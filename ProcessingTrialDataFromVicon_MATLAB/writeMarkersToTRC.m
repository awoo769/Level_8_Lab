function err = writeMarkersToTRC(trcfile, Markers, MLabels, Rate, Frames, Time, Units)
% Write 3D Markers trajectories (real or virtual) to a .trc file                        
% USAGE: error = writeMarkersToTRC(trcFile, Markers, MLabels, Rate, Frames, Time, Units)
%
% This function was written by Ajay Seth at Stanford University in 2007, and is
% being used alongside other code by Duncan Bakke at the Auckland (edited)

err = 0;
% Generate the header for the .trc file
fid = fopen(trcfile, 'wt');

fprintf(fid, 'PathFileType\t4\t(X/Y/Z)\t%s\n', trcfile);
fprintf(fid, 'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n');
fprintf(fid, '%f\t%f\t%d\t%d\t%s\t%f\t%d\t%d\n', ...
    Rate, Rate, length(Frames), length(MLabels), Units, Rate, Frames(1), Frames(end));
fprintf(fid, 'Frame#\tTime');
for I = 1:length(MLabels),
    fprintf(fid,'\t%s\t\t', MLabels{I});
end
fprintf(fid, '\n\t');
for I = 1:length(MLabels),
    fprintf(fid,'\tX%i\tY%i\tZ%i', I,I,I);
end
fprintf(fid, '\n\n');

%Print all frames, timestamps, and markers
for i = 1:length(Frames)
   fprintf(fid,'%i\t%f\t',Frames(i),Time(i));
   for j = 1:length(Markers)
      fprintf(fid,'%f\t%f\t%f\t',Markers(j).x(i),Markers(j).y(i),Markers(j).z(i));
   end
   if i<length(Frames)
   fprintf(fid,'\n');
   end
end

fclose(fid);

end