function mergedParams = MergeParams(baseParams,updatedParams)

bNames = fieldnames(baseParams);
uNames = fieldnames(updatedParams);
diffNames = setdiff(bNames,uNames);

mergedParams = updatedParams;
for i=1:length(diffNames)
    if isfield(baseParams,diffNames{i})
        mergedParams.(diffNames{i}) = baseParams.(diffNames{i});
    end
end