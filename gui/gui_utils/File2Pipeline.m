function [pipeline,baseFile] = File2Pipeline(filePath)
% [pipeline, baseFile] = File2Pipeline(filePath)
%
%   Finds the pipeline which led to this file name.
% 
% MNA 2/19/2020 

str = regexp(filePath,'[\/\\_.]','split');

pipeline = {};
pipeCounter = 0;

for i=1:length(str)
    [~,~,module] = Filename('',str{i});
    if ~isempty(module) 
        if ~contains(module,'inmemory')
            pipeCounter = pipeCounter + 1;
            pipeline{pipeCounter} = module; %#ok<AGROW>
        end
    end
end

% get keyword for first module
if ~isempty(pipeline)
    [~,keyword] = Filename('',pipeline{1});
    ix = strfind(filePath,keyword);
else
    ix = [];
end

if ~isempty(ix)
    baseFile = filePath(1:ix-2);
else
    baseFile = filePath;
end