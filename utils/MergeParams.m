function mergedParams = MergeParams(smallParams,bigParams)

sNames = fieldnames(smallParams);
bNames = fieldnames(bigParams);
diffNames = setdiff(sNames,bNames);

mergedParams = bigParams;
for i=1:length(diffNames)
    if isfield(smallParams,diffNames{i})
        mergedParams.(diffNames{i}) = smallParams.(diffNames{i});
    end
end