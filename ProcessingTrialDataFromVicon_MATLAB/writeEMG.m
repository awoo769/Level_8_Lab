function writeEMG(EMGdata,headers,emgFilename)
%Writes EMG mot files. Based on a script originally written by Ajay Seth.

emgFile = fopen(emgFilename, 'w');
fprintf(emgFile, 'Normalized EMG Linear Envelopes\n');
fprintf(emgFile, 'nRows=%d\n', size(EMGdata,1));
fprintf(emgFile, 'nColumns=%d\n\n', size(EMGdata,2));
fprintf(emgFile, 'endheader\n');

for i=1:length(headers)
    if i == 1
        fprintf(emgFile, '%s\t', headers{i});
    else
        fprintf(emgFile, '%s\t', headers{i});
    end
end
fprintf(emgFile, '\n');

for i=1:size(EMGdata,1)
    fprintf(emgFile, '%4.3f\t', EMGdata(i,1));
	fprintf(emgFile, '%10.6f\t',EMGdata(i,2:end));
	fprintf(emgFile, '\n');
end

fclose(emgFile);
return;
