function err = changeLoadXMLFile(filename,title,model,directory,cutoffFreq)
% This function takes the external loads xml file specified by filename,
% and sets the .mot files to those specified by title, and writes to a new
% file of the two strings combined, i.e. "Walk1ExternalLoads.xml"ID
%% Rewrite External Load XML file for new title
err = 0;
docNode = xmlread(filename);
% Hardcode GRF mot location/name
motString = strcat(directory,'\',model,'\',title,'\',title,'.mot');
IKstring = strcat(title,'IKResults.mot');
%% Get Hierarchy Access
exL = docNode.getElementsByTagName('ExternalLoads');
exLChild = exL.item(0);
dataf = exLChild.getElementsByTagName('datafile');
datafileChild = dataf.item(0);
exLmodel = exLChild.getElementsByTagName('external_loads_model_kinematics_file');
exLmodelChild = exLmodel.item(0);
cutoff = exLChild.getElementsByTagName('lowpass_cutoff_frequency_for_load_kinematics');
cutoffChild = cutoff.item(0);
%% Write new ones
datafileChild.getFirstChild.setData(motString);
% exLmodelChild.getFirstChild.setData(IKstring); This should be
% unnecessary.
cutoffChild.getFirstChild.setData(num2str(cutoffFreq));
%% Write file
newfilename = strcat(title, filename);
xmlwrite(newfilename,docNode);
end