function xmlShorten(filePath)
% Read file.
fId = fopen(filePath, 'r');
fileContents = fread(fId, '*char')';
fclose(fId);
% Write new file.
fId = fopen(filePath, 'w');
% Remove extra lines.
fwrite(fId, regexprep(fileContents, '\n\s*\n', '\n'));
fclose(fId);
end