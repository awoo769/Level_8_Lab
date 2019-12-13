function [headers,data,sampleFreq] = readEMGMOT(MOTfilename)
MOTfile = fopen(MOTfilename, 'r');

if MOTfile == -1
    error(['fopen cannot access ', MOTfilename])
end

nameLine = fgetl(MOTfile);

frequencyLine = fgetl(MOTfile);
sampleFreq = sscanf(frequencyLine, '%f');
descriptionLine = fgetl(MOTfile);
headerLine = fgetl(MOTfile);
headers = strsplit(headerLine,'\t');
unitLine = fgetl(MOTfile);
curLine = fgetl(MOTfile);
i = 0;
while ~isempty(curLine)
    i=i+1;
    data(i,:) = sscanf(curLine, '%f', length(headers));
    curLine = fgetl(MOTfile);
end
end