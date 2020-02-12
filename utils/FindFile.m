function inputFile = FindFile(inputFile)

str = which(inputFile);
if ~isempty(str)
    inputFile = str;
end