function [newMarkers, newFrames] = trimTRC(oldMarkers,oldFrames,inds)

for i = 1:length(oldMarkers)
    newMarkers(i).x = oldMarkers(i).x(inds(1):inds(2));
    newMarkers(i).y = oldMarkers(i).y(inds(1):inds(2));
    newMarkers(i).z = oldMarkers(i).z(inds(1):inds(2));
end

newFrames = oldFrames(inds(1):inds(2),:);

end