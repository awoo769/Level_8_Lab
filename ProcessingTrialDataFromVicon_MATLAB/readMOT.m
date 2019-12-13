function [headers,data] = readMOT(MOTfilename)
MOTfile = fopen(MOTfilename, 'r');

if MOTfile == -1
    error(['fopen cannot access ', MOTfilename])
end

nameLine = fgetl(MOTfile);
versionLine = fgetl(MOTfile);

rowsLine = fgetl(MOTfile);
numRows = sscanf(rowsLine, 'nRows=%f');

columnsLine = fgetl(MOTfile);
numColumns = sscanf(columnsLine, 'nColumns=%f');

degreesLine = fgetl(MOTfile);
endheaderLine = fgetl(MOTfile);

clearvars -except MOTfile numColumns numRows

headerLine = fgetl(MOTfile);
headers = strsplit(headerLine);

data = fscanf(MOTfile, '%f', [numColumns, numRows])';

fclose(MOTfile);
end