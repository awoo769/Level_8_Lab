function prepareTrialFromVicon(model, trial, directory, inputDirectory)
% prepareTrialFromVicon: A function to condition and collate trial data and
% setup all necessary OpenSim analysis xmls.
% Inputs:   model = Name of subject, assuming model file is "Subject.osim".
%           trial = Name of motion capture trial.
%           directory = Location of output.
%           inputDirectory = Location of input files.
%% Initial Setup/Names
% if ~exist('SSTrials','var') %If using H Trials
%    populateSSTrials;
% end
badEMGTrials = {'SAFIST015_SS21_20Jun_ss_035ms_02','SAFIST015_SS21_20Jun_fast_075ms_02','SAFIST015_SS42_20Jun_ss_035ms_01','SAFIST015_SS42_20Jun_fast_055ms_01','SAFIST015_SS52_ss_04ms_02','SAFIST015_SS52_fast_07ms_01','SS77_SAFIST015_18Jun_fast_04ms_02','SAFIST015_19Jun_SS90_ss_035ms_01','SAFIST015_19Jun_SS90_fast_055ms_01','_12Mar_ss_12ms_01'};
badEMG = 0;
recalculateCOP = 1;
if any(contains(badEMGTrials,trial))
    badEMG = 1;
end

%Identify files from Vicon Export to read
trcFilename = fullfile(inputDirectory,strcat(trial,'.trc'));
motFilename = fullfile(inputDirectory,strcat(trial,'.mot'));
emgFilename = fullfile(inputDirectory,strcat(trial,'_EMG.mot'));

if ~isfile(trcFilename)
    trcFilename = fullfile(inputDirectory,strcat(model,trial,'.trc'));
end

if ~isfile(motFilename)
    motFilename = fullfile(inputDirectory,strcat(model,trial,'.mot'));
end

if ~isfile(emgFilename)
    emgFilename = fullfile(inputDirectory,strcat(model,trial,'_EMG.mot'));
    if ~isfile(emgFilename)
        fprintf('No EMG for subject %s.\n',model)
        badEMG = 1;
end

% txtfilename = fullfile(inputDirectory,strcat(trial,'.txt')); No Gait Events in SS set

mkdir('Output',model);
mkdir(strcat('Output','\',model),trial);
newFolder = fullfile(pwd,'Output',model,trial);
IKfilename = 'IKSetup.xml';
IDfilename = 'IDSetup.xml';
exLoadsFilename = 'ExternalLoads.xml';
MuscleAnalysisfilename = 'MuscleAnalysisSetup.xml';
MuscleForceDirectionfilename = 'MuscleForceDirectionSetup.xml';
addpath(genpath([pwd '\xmlTemplates']));

%% Pull in Exported Vicon Files, Identify time range of interest
% (Note: This approach differs with regard to available event data.)
[frames, markerNames, markers, datarate] = readTRC(trcFilename);
% [left, right] = readEventsTXT(txtfilename);
disp(model);
disp(trial);

if ~badEMG
    [EMGheaders,EMGdata,EMGfreq] = readEMGMOT(emgFilename);
end
[GRFheaders,fullGRFdata] = readMOT(motFilename);

% Info Assignment
% trialData = HTrials(find(strcmp({HTrials.name}, model)==1));
% steps = trialData.steps;
% plates = trialData.plates;

if contains(model,'SS') || contains(model,'AB') %If SS or AB, recorded at Millenium.
    steps = {'l','r'};
    plates = [1,2];
end

% If overground, this ensures correct time limits.
% if strcmp(steps{1},'L')
%     timerange(1) = min(right.off);
%     timerange(2) = max(left.strike);
% else
%     timerange(1) = min(left.off);
%     timerange(2) = max(right.strike);
% end
%

timerange(1) = round(max(frames(1,2),0)+0.020,3);
timerange(2) = frames(end,2);

indexStart = find(frames(:,2) == timerange(1));
indexEnd = find(frames(:,2) == timerange(2));

framerange(1) = frames(indexStart,1);
framerange(2) = frames(indexEnd,1);


%% IK File
[trimmedMarkers, trimmedFrames] = trimTRC(markers,frames,[indexStart indexEnd]);
[goodMarkers,goodMarkerNames,badMarkerNames] = removeBadMarkers(trimmedMarkers,markerNames);

writeMarkersToTRC(fullfile(newFolder,strcat(trial,'.trc')),goodMarkers,goodMarkerNames,datarate,trimmedFrames(:,1),trimmedFrames(:,2),'mm');

%Note that this function edits the xml file. This is better done using the
%OpenSim APIs if you can.
IKerr = changeIKXMLFile(IKfilename,trial,timerange,goodMarkerNames,badMarkerNames,model,directory); %#ok<NASGU>

fullIKfilename = strcat(trial,IKfilename);
xmlShorten(fullIKfilename);
% Move File
movefile(fullIKfilename, newFolder);

%% ID Files

%Define rate of GRF capture
GRFrate = (length(fullGRFdata)-1)/(fullGRFdata(end,1)-fullGRFdata(1,1));

% Get Original Vertical Forces
originalFys = fullGRFdata(:,find(not(~contains(GRFheaders, 'vy'))));

% Condition GRFData
[b, a] = butter(4, (10/(GRFrate/2)));
newGRFdata(:,1) = fullGRFdata(:,1);
for i = 2:length(GRFheaders)
    newGRFdata(:,i) = filtfilt(b, a, fullGRFdata(:,i));
end

% Re-Zero GRFs
filterPlate = reZeroFilter(originalFys);

% Rezero everything except for centre of pressure
for i = 2:length(GRFheaders)
    % Force plate 1, values which are not CoP - WHY NOT JUST SEARCH IF 1 or
    % 2?
    if isempty(strfind(GRFheaders{i},'p')) && i<=(length(GRFheaders)/2) %this relies on centre of pressure columns containing the letter 'p' and NO OTHER COLUMNS doing so.
        newGRFdata(:,i) = filterPlate(:,1).*newGRFdata(:,i);
        
    % Force plate 2, values which are not CoP
    elseif isempty(strfind(GRFheaders{i},'p'))
        newGRFdata(:,i) = filterPlate(:,2).*newGRFdata(:,i);
    end
end

if recalculateCOP
    %Define for recaluclating CoP
    xoffset = [0.2385 0.7275];
    yoffset = [0 0];
    vzInds = find(contains(GRFheaders,'vy')); %OpenSIM COORDS
    pxInds = find(contains(GRFheaders,'px'));
    pyInds = find(contains(GRFheaders,'pz'));
    
    %Back Calculate Moment Measurements
    for i = 1:length(plates)
        sideInds = find(contains(GRFheaders,num2str(i)));
        fZ(i,:) = fullGRFdata(:,intersect(vzInds,sideInds));
        pX(i,:) = fullGRFdata(:,intersect(pxInds,sideInds));
        pY(i,:) = fullGRFdata(:,intersect(pyInds,sideInds));
        oldmY(i,:) = (xoffset(i)-pX(i,:)).*fZ(i,:);
        oldmX(i,:) = (yoffset(i)+pY(i,:)).*fZ(i,:);
    end
    
    for i = 1:length(plates)
        mY(i,:) = filtfilt(b,a,oldmY(i,:));
        mX(i,:) = filtfilt(b,a,oldmX(i,:));
    end
    
    for i = 1:length(plates)
        mY(i,:) = filterPlate(:,i)'.*mY(i,:);
        mX(i,:) = filterPlate(:,i)'.*mX(i,:);
    end
    
    %Recalculate CoP with Filtered forces and moments
    for i = 1:length(plates)
        sideInds = find(contains(GRFheaders,num2str(i)));
        newfZ = newGRFdata(:,intersect(vzInds,sideInds));
        newpX(i,:) = xoffset(i)-(mY(i,:)./newfZ');
        newpY(i,:) = yoffset(i)+(mX(i,:)./newfZ');
        
        for j=1:length(newpX(i,:))
            if isnan(newpX(i,j))
                newpX(i,j)=0;
                newpY(i,j)=0;
            end
        end
        
        newGRFdata(:,intersect(pxInds,sideInds)) = newpX(i,:);
        newGRFdata(:,intersect(pyInds,sideInds)) = newpY(i,:);
        
        figure;
        plot(pX(i,:),pY(i,:),'*');
        hold on
        plot(newpX(i,:),newpY(i,:),'x');
        fprintf('Just associated new CoP for plate %i\n',i)
    end
    
end

GRFdata = newGRFdata(find(newGRFdata(:,1)==timerange(1)):find(newGRFdata(:,1)==timerange(2)),:);

newHeaders = fixGRFheaders(GRFheaders,steps,plates);

writeMOT(fullfile(newFolder,strcat(trial,'.mot')),newHeaders,GRFdata);

IDerr = changeIDXMLFile(IDfilename,trial,timerange,model,directory,10); %#ok<NASGU>
xmlShorten(strcat(trial,IDfilename));

ExLerr = changeLoadXMLFile(exLoadsFilename,trial,model,directory,10); %#ok<NASGU>
xmlShorten(strcat(trial,exLoadsFilename));

% % Move Files
fullIDfilename = strcat(trial,IDfilename);
movefile(fullIDfilename, newFolder);
fullexLoadsFilename = strcat(trial,exLoadsFilename);
movefile(fullexLoadsFilename, newFolder);

%% EMG Processing
if ~badEMG
    EMGenv = envelopeEMG(EMGdata,EMGfreq);
    if strfind(EMGheaders{1},'Frame') && strfind(EMGheaders{2},'Frame')
        EMGenv(:,1:2) = EMGdata(:,1:2);
    end
    emgDelay = 0.02; % 2 frames (@100 Hz) or 4 frames (@200 Hz)
    
    frameOffset = emgDelay/(1/datarate);
    
    EMGstart = find(EMGenv(:,1)==framerange(1)-frameOffset);
    EMGend = find(EMGenv(:,1)==framerange(2)-frameOffset);
    EMGtime = [(timerange(1)-emgDelay):(1/EMGfreq):(timerange(2)-emgDelay)]';
    
    clippedEMG = EMGenv(EMGstart:EMGend,:);
    EMG = [EMGtime clippedEMG(:,3:end)];
    EMGlabels = {'time' EMGheaders{3:end}};
    
    writeEMG(EMG,EMGlabels,fullfile(newFolder,strcat(trial,'_EMG.mot')));
end

%% Muscle Analysis Files
MAerr = changeMuscleAnalysisXMLFile(MuscleAnalysisfilename,trial,timerange,model);
xmlShorten(strcat(trial,MuscleAnalysisfilename));

MAerr = changeMuscleForceDirectionXMLFile(MuscleForceDirectionfilename,trial,timerange,model);
xmlShorten(strcat(trial,MuscleForceDirectionfilename));

% Move Files
fullMALoadsFilename = strcat(trial,MuscleAnalysisfilename);
movefile(fullMALoadsFilename, newFolder);
fullMFDLoadsFilename = strcat(trial,MuscleForceDirectionfilename);
movefile(fullMFDLoadsFilename, newFolder);
end