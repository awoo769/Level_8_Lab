function writeMOT(filename, headers, data)
MOTfile = fopen(filename, 'w');	

fprintf(MOTfile, 'name %s\n', filename);
fprintf(MOTfile, 'datacolumns %d\n', size(data,2));
fprintf(MOTfile, 'datarows %d\n', size(data,1));
fprintf(MOTfile, 'range %f %f\n', min(data(:,1)), max(data(:,1)));
fprintf(MOTfile, 'endheader\n\n');

fprintf(MOTfile, '\t');

for i=1:length(headers)
	fprintf(MOTfile, '%s\t', headers{i});
end
fprintf(MOTfile, '\n');

for i=1:size(data,1)
	fprintf(MOTfile, '%20.8f\t', data(i,:));
	fprintf(MOTfile, '\n');
end

fclose(MOTfile);
end