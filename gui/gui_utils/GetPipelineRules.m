function pipeRules = GetPipelineRules()
% GetPipelineRules() Returns available modules and their connection rules
%
%   Output is a stucture 'pipeRules' with the following fields.
%       before: 2D array of 1s and 0s indicating before connections
%       after: 2D array of 1s and 0s indicating after connections
%       modules: cell array of module names
%
% MNA 2/19/2020

% availableModules = {'findblinkframes';''
%                     'trimvideo';
%                     'removestimuli';
%                     'gammacorrect';
%                     'bandpassfilter';
%                     'stripanalysis';
%                     'pixel2degree';
%                     'degree2pixel';
%                     'filtereyeposition';
%                     'makereference';
%                     'rereference';
%                     'findsaccadesanddrifts'};
listing = dir([fileparts(which('StripAnalysis')) filesep '*.m']);
availableModules = arrayfun(@(x) lower(regexp(x{1},'.*[^\.m]','match')),{listing.name}');
                
% also add 'none'
availableModules{end+1} = 'none';

n = length(availableModules);
beforeMatrix = zeros(n);
afterMatrix = zeros(n);

for i=1:n
    [~,~,before,after] = GetDefaults(availableModules{i});
    
    for j=1:length(before)
        thisModuleIx = contains(availableModules,before{j});
        beforeMatrix(i,thisModuleIx) = 1;
    end
    
    for j=1:length(after)
        thisModuleIx = contains(availableModules,after{j});
        afterMatrix(i,thisModuleIx) = 1;
    end
end

pipeRules.before = beforeMatrix;
pipeRules.after = afterMatrix;
pipeRules.modules = availableModules;
