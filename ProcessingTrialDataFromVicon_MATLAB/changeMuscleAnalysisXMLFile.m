function err = changeMuscleAnalysisXMLFile(filename,title,timerange,model)
%% Rewrite Analysis Setup XML file for new title
err = 0;
docNode = xmlread(filename);

% Note that any attribute not changed within this function must be changed
% in the original template in "xmlTemplates"

%% Get Hierarchy Access
AnalyzeTool = docNode.getElementsByTagName('AnalyzeTool');
AnalyzeToolChild = AnalyzeTool.item(0);

resDirectory = AnalyzeToolChild.getElementsByTagName('results_directory');
resDirectoryChild = resDirectory.item(0);

model_file =  AnalyzeToolChild.getElementsByTagName('model_file');
model_fileChild = model_file.item(0);

initialTime =  AnalyzeToolChild.getElementsByTagName('initial_time');
initialTimeChild = initialTime.item(0);

finalTime =  AnalyzeToolChild.getElementsByTagName('final_time');
finalTimeChild = finalTime.item(0);

externalLoadsFile =  AnalyzeToolChild.getElementsByTagName('external_loads_file');
exLoadsFileChild = externalLoadsFile.item(0);

CoordinateFile =  AnalyzeToolChild.getElementsByTagName('coordinates_file');
CoordinateFileChild = CoordinateFile.item(0);

analysisSet =  AnalyzeToolChild.getElementsByTagName('AnalysisSet');
analysisSetChild = analysisSet.item(0);

objectsSet =  analysisSetChild.getElementsByTagName('objects');
objectsSetChild = objectsSet.item(0);

% Muscle Analysis Settings

muscleAnalysis =  objectsSetChild.getElementsByTagName('MuscleAnalysis');
muscleAnalysisChild = muscleAnalysis.item(0);

muscleAnalysisStartTime =  muscleAnalysisChild.getElementsByTagName('start_time');
muscleAnalysisStartTimeChild = muscleAnalysisStartTime.item(0);

muscleAnalysisEndTime =  muscleAnalysisChild.getElementsByTagName('end_time');
muscleAnalysisEndTimeChild = muscleAnalysisEndTime.item(0);

%% Set New Directory, Filenames, and number inputs

%newAnalyzeToolName = [model title];
%AnalyzeToolChild.setAttribute('name',newAnalyzeToolName);

resDirectoryChild.getFirstChild.setData('.\');

modelFileName = strcat(model,'.osim');
model_fileChild.getFirstChild.setData(modelFileName);

CoordinateFileName = strcat(title,'IKResults.mot');
CoordinateFileChild.getFirstChild.setData(CoordinateFileName);

externalLoadsFile = strcat(title,'ExternalLoads.xml');
exLoadsFileChild.getFirstChild.setData(externalLoadsFile);

%Set Start and End times for all tools.
startTime = num2str(timerange(1));
endTime = num2str(timerange(2));
initialTimeChild.getFirstChild.setData(startTime);
finalTimeChild.getFirstChild.setData(endTime);
muscleAnalysisStartTimeChild.getFirstChild.setData(startTime);
muscleAnalysisEndTimeChild.getFirstChild.setData(endTime);
%% Write file
newfilename = strcat(title, 'MuscleAnalysisSetup.xml');
xmlwrite(newfilename,docNode);
end