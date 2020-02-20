function [tf,startAt,baseFile,skippedIx] = IsValidPipeline(pipeline,inputFile)
% [tf,startAt,baseFile,skippedIx] = IsValidPipeline(pipeline, inputFile) 
% 
%   Checks if pipeline breaks any connection rules
%
%   Input is a cell array of module names in the order that they are to be
%   executed, i.e. pipeline. The output is a logical.
%
% MNA 2/19/2020

% for robustness
pipeline = lower(pipeline);

% by default we start at first module of the pipeline when running
startAt = 1;

% get the rules first
pipeRules = GetPipelineRules();

if nargin < 2 
    inputFile = [];
end

if ~isempty(inputFile)
    [previousPipeline, baseFile] = File2Pipeline(inputFile);
    
    if ~isempty(previousPipeline)
        % check if this file was partially processed using current pipeline
        ix = contains(pipeline, previousPipeline);
        lastModuleNo = find(ix==1,1,'last');
        skippedIx = find(ix(1:lastModuleNo) == 0);
        skippedModuleCount = length(skippedIx);
        
        % normally skippedModuleCount should be zero if file was processed
        % contigously but one of the modules' (FindBlinkFrames) output go
        % on a separate file than the primary output of that module. So we
        % check here how many of those "skips" are due to this reason, and
        % we count down the skipped module count.
        if skippedModuleCount>0
            for i=1:length(skippedIx)
                if contains(pipeline{skippedIx(i)},'findblinkframes')
                    skippedModuleCount = skippedModuleCount - 1;
                end
            end
        end
        
        if skippedModuleCount<=0
            startAt = lastModuleNo+1;
        else
            % if this file was processed with a different pipeline, check if we
            % can continue processing with the current pipeline. To do that, we
            % add last module of the input file to the beginning of current
            % pipe and the rest of this function will do the check.
            pipeline = [previousPipeline(end); pipeline];

        end
    else
        skippedIx = [];
    end
else
    baseFile = [];
end


% num of modules in the current pipeline
n = length(pipeline);

tf = true;

for i=1:n
    % this module
    current = contains(pipeRules.modules,pipeline{i});
    
    % before test
    if i==1
        % skip before test
    else
        prev = contains(pipeRules.modules,pipeline{i-1});
        if ~pipeRules.before(current,prev)
            fprintf('%s cannot come before %s.\n',pipeline{i-1},pipeline{i})
            tf = false;
            break;
        end
    end
    
    % after test
    if i==n
        % skip after test
    else
        next = contains(pipeRules.modules,pipeline{i+1});
        if ~pipeRules.after(current,next)
            fprintf('%s cannot come after %s.\n',pipeline{i+1},pipeline{i})
            tf = false;
            break;
        end
    end
end

