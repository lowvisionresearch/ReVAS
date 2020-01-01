function inputFile = FindFile(inputFile)

str = which(inputFile);
if ~isempty(str)
    [filepath,~,~] = fileparts(str);
    inputFile = [filepath filesep inputFile];
end