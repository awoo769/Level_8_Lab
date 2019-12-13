function timeRange = changeIKXMLFile(filename,title,timerange,goodMarkerNames,badMarkerNames,model,directory)
%% Rewrite IK Setup XML file for new title
err = 0;
docNode = xmlread(filename);

%% Get Hierarchy Access
IKTool = docNode.getElementsByTagName('InverseKinematicsTool');
IKToolChild = IKTool.item(0);

resDirectory = IKToolChild.getElementsByTagName('results_directory');
resDirectoryChild = resDirectory.item(0);

inputDirectory = IKToolChild.getElementsByTagName('input_directory');
inputDirectoryChild = inputDirectory.item(0);

model_file = IKToolChild.getElementsByTagName('model_file');
model_fileChild = model_file.item(0);

IKTaskSet = IKToolChild.getElementsByTagName('IKTaskSet');
IKTaskSetChild = IKTaskSet.item(0);

IKTaskSetObjects = IKTaskSetChild.getElementsByTagName('objects');
IKTaskSetObjectsChild = IKTaskSetObjects.item(0);

IKMarkerTasks = IKTaskSetObjectsChild.getElementsByTagName('IKMarkerTask');
numMarkers = IKMarkerTasks.getLength();

marker_file = IKToolChild.getElementsByTagName('marker_file');
marker_fileChild = marker_file.item(0);

time_range = IKToolChild.getElementsByTagName('time_range');
time_rangeChild = time_range.item(0);

output_motion_file = IKToolChild.getElementsByTagName('output_motion_file');
output_motion_fileChild = output_motion_file.item(0);

%% Set New Directory, Filenames, and number inputs
resDirectoryChild.getFirstChild.setData('.\');
inputDirectoryChild.getFirstChild.setData('.\');

%OpenSim model name
modelFileName = strcat(directory,'\',model,'\',title,'\',model,'.osim');
model_fileChild.getFirstChild.setData(modelFileName);

%Hardcode input trc
markerFileName = strcat(directory,'\',model,'\',title,'\',title,'.trc');
marker_fileChild.getFirstChild.setData(markerFileName);

%Plain output name (for local results)
outputFileName = strcat(title,'IKResults.mot');
output_motion_fileChild.getFirstChild.setData(outputFileName);

%Time Range
timeRange = strcat(num2str(timerange(1)), {' '}, num2str(timerange(2)));
time_rangeChild.getFirstChild.setData(timeRange);

%% Remove any absent markers, Set weighting for bony landmarks
for i = 0:numMarkers-1
   currentMarker = IKMarkerTasks.item(i);
   currentMarkerName = char(currentMarker.getAttribute('name'));
   apply = currentMarker.getElementsByTagName('apply');
   applyChild = apply.item(0);
   weight = currentMarker.getElementsByTagName('weight');
   weightChild = weight.item(0);
   if (ismember(currentMarkerName,goodMarkerNames))&&(~ismember(currentMarkerName,badMarkerNames))
       applyChild.getFirstChild.setData('true');
   else
       applyChild.getFirstChild.setData('false');
   end
   if ismember(currentMarkerName,{'LMMAL','RMMAL','LLMAL','RLMAL','LASI','RASI','LPSI','RPSI'})
       weightChild.getFirstChild.setData(num2str(10));
   else
       weightChild.getFirstChild.setData(num2str(1));
   end
end

%% Write file
newfilename = strcat(title, filename);
xmlwrite(newfilename,docNode);
end