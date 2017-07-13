function SkipRemoveStimuli(inputVideoPath, parametersStructure)
%SKIP REMOVE STIMULI Run instead of |RemoveStim| to skip this step in the
%pipeline while maintaining internal consistency.
%   Run instead of |RemoveStim| to skip this step in the pipeline while
%   maintaining internal consistency.

outputVideoPath = [inputVideoPath(1:end-4) '_nostim' inputVideoPath(end-3:end)];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
else
    RevasWarning(['SkipRemoveStimuli() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
end

%% Copy and rename video

copyfile(inputVideoPath, outputVideoPath);

end

