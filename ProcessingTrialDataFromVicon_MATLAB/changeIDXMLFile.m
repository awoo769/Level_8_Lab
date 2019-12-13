function err = changeIDXMLFile(filename,title,timerange,model,directory,cutoffFreq)
%% Rewrite ID Setup XML file for new title
err = 0;
docNode = xmlread(filename);

if nargin <5
    cutoffFreq = 6;
end

%% Get Hierarchy Access
IDTool = docNode.getElementsByTagName('InverseDynamicsTool');
IDToolChild = IDTool.item(0);

resDirectory = IDToolChild.getElementsByTagName('results_directory');
resDirectoryChild = resDirectory.item(0);

inputDirectory = IDToolChild.getElementsByTagName('input_directory');
inputDirectoryChild = inputDirectory.item(0);

model_file = IDToolChild.getElementsByTagName('model_file');
model_fileChild = model_file.item(0);

time_range = IDToolChild.getElementsByTagName('time_range');
time_rangeChild = time_range.item(0);

exLoads_file = IDToolChild.getElementsByTagName('external_loads_file');
exLoads_fileChild = exLoads_file.item(0);

coords_file = IDToolChild.getElementsByTagName('coordinates_file');
coords_fileChild = coords_file.item(0);

filterFrequency = IDToolChild.getElementsByTagName('lowpass_cutoff_frequency_for_coordinates');
filterFrequencyChild = filterFrequency.item(0);

output_gen_force_file = IDToolChild.getElementsByTagName('output_gen_force_file');
output_gen_force_fileChild = output_gen_force_file.item(0);

%% Set New Directory, Filenames, and number inputs

IDToolChild.setAttribute('name',model)

%Local directory
inputDirectoryChild.getFirstChild.setData('.\');
resDirectoryChild.getFirstChild.setData('.\');

%OpenSim Model
modelFileName = strcat(directory,'\',model,'\',title,'\',model,'.osim');
model_fileChild.getFirstChild.setData(modelFileName);

%Time Range
timeRange = strcat(num2str(timerange(1)), {' '}, num2str(timerange(2)));
time_rangeChild.getFirstChild.setData(timeRange);

externalLoadsFile = strcat(title,'ExternalLoads.xml');
exLoads_fileChild.getFirstChild.setData(externalLoadsFile);

coordsFile = strcat(title,'IKResults.mot');
coords_fileChild.getFirstChild.setData(coordsFile);

filterFrequencyChild.getFirstChild.setData(num2str(cutoffFreq));

outputFileName = strcat(title,'IDResults.sto');
output_gen_force_fileChild.getFirstChild.setData(outputFileName);

%% Write file
newfilename = strcat(title, filename);
xmlwrite(newfilename,docNode);
end