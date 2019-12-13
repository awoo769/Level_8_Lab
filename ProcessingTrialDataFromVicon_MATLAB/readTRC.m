function [frames, markerNames, markers, dataRate] = readTRC(TRCfilename)

TRCfile = fopen(TRCfilename, 'r');
if TRCfile == -1
    error(['unable to open ', TRCfilename])
end

topLine = fgetl(TRCfile);
statsheaderLine = fgetl(TRCfile);
statsLine = fgetl(TRCfile);
stats = sscanf(statsLine, '%f',4);
dataRate = stats(1);
numFrames = stats(3);
numMarkers = stats(4);
numColumns=2+(3*numMarkers);

markerLine = fgetl(TRCfile);
markerNames = strsplit(markerLine);

if isempty(markerNames{end}) %Restrict to marker names only
    markerNames = markerNames(3:end-1);
else
    markerNames = markerNames(3:end);
end

if strfind(markerNames{1},':') %Remove Vicon Model Prefix if present
    startInd = strfind(markerNames{1},':')+1;
    markerNames = cellfun(@(x) x(startInd:end),markerNames,'UniformOutput',false);
end

notData = 1;
while notData
    newLine = fgetl(TRCfile);
    if strfind(newLine,'X1')
        notData = 0;
    end
end

j = 0;
numBlanks = 0;
while ~feof(TRCfile)
    j = j+1;
    dataline = fgetl(TRCfile);
    datavalues = sscanf(dataline,'%f');
    if isempty(dataline)
        if numBlanks < 1
            numBlanks = numBlanks + 1;
            j = j-1;
            continue
        else
            break
        end
    end
    [~,matches] = strsplit(dataline);
    blanksStrings = regexprep(matches, '\t\t\t', 'missing');
    
    curOffset = 0;
    origLength = length(datavalues);
    
    missingIndex = strfind(blanksStrings, 'missing');
    missingIndices = find(not(cellfun('isempty', missingIndex)));
    
    for i = 1:length(missingIndices)
        numMissing = length(strfind(blanksStrings{missingIndices(i)},'missing'));
        blankEntries = zeros(numMissing*3,1);
        if missingIndices(i) == origLength
            datavalues = [datavalues; blankEntries];
        else
            datavalues = [datavalues(1:missingIndices(i)+curOffset); blankEntries; datavalues(missingIndices(i)+curOffset+1:end)];
        end
        curOffset = curOffset + length(blankEntries);
    end
    data(j,:) = datavalues;
end
fclose(TRCfile);
clearvars -except data numMarkers markerNames dataRate

frames = data(:,1:2);

for m=1:numMarkers
    markers(m).x = data(:,2+(3*(m-1))+1);
    markers(m).y = data(:,2+(3*(m-1))+2);
    markers(m).z = data(:,2+(3*(m-1))+3);
end
end