function newHeaders = fixGRFheaders(GRFheaders,steps,plates)
%fixGRFheaders: takes Vicon output headers and converts them to logical
%ones (that also fit the existing External Load file template)
leftPlate = num2str(plates(find(strcmpi(steps,'l'))));
rightPlate = num2str(plates(find(strcmpi(steps,'r'))));

for i=1:length(GRFheaders)
    if strfind(GRFheaders{i},leftPlate)
        numInd = strfind(GRFheaders{i},leftPlate);
        cleanHeader = [GRFheaders{i}(1:numInd-1) GRFheaders{i}(numInd+1:end)];
        newHeaders{i} = ['L_' cleanHeader];
    elseif strfind(GRFheaders{i},rightPlate)
        numInd = strfind(GRFheaders{i},rightPlate);
        cleanHeader = [GRFheaders{i}(1:numInd-1) GRFheaders{i}(numInd+1:end)];
        newHeaders{i} = ['R_' cleanHeader];
    else
       newHeaders{i} = GRFheaders{i};
    end
end
end

