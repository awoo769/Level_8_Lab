function [filterPlate] = reZeroFilter(originalFys)
%reZeroForceData: Resets all values that were originally zero to zero.

for i = 1:length(originalFys(1,:))
    positiveFy = originalFys(:,i) > 20;
    
    % Finds index of 1's, only if there are at least 9 more 1's ahead of
    % it?
    trueInd = strfind(positiveFy',[1 1 1 1 1 1 1 1 1 1]);
    
    % But then adds 10 on to this - cancelling out what was just done?
    extraInd = trueInd(1:end-1) + 10;
    
    filterPlate(:,i) = zeros(length(originalFys(:,i)),1);
    filterPlate(trueInd,i) = 1;
    filterPlate(extraInd,i) = 1;
end