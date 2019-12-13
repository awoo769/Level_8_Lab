function [goodMarkers,goodMarkerNames, badMarkerNames] = removeBadMarkers(trimmedMarkers,markerNames)
%check for gaps
badInds = [];
for i = 1:length(trimmedMarkers)
   if length(find(abs(trimmedMarkers(i).x) == 0)) > 10
      badInds =  [badInds i];
   end
end

%check for duplicates
[~,uniqueID] = unique(markerNames);
duplicateMarkers = markerNames;
duplicateMarkers(uniqueID) = [];

if ~isempty(duplicateMarkers)
    for i = 1:length(duplicateMarkers)
        dupeMarkerInds = find(contains(markerNames,duplicateMarkers{i}));
        badInds = [badInds dupeMarkerInds(2:end)];
    end
end

%remove all bad markers
onesvector = ones(length(trimmedMarkers),1);
onesvector(badInds) = 0;
goodMarkers = trimmedMarkers(find(onesvector==1));
goodMarkerNames = markerNames(find(onesvector==1));
badMarkerNames = markerNames(find(onesvector~=1));    
end