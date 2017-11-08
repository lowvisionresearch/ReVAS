function refinedFrame = FineRef(coarseRefFrame, inputVideoPath, parametersStructure)
%FINE REF  Generate a better reference frame.
%   The function alternates between StripAnalysis and MakeMontage,
%   alternating between generating positions and generating the reference
%   frames that result from those positions.
%   

%% Handle overwrite scenarios.

% No files are saved by this function so no overwrite checking necessary.

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'numberOfIterations')
    numberOfIterations = 1;
else
    numberOfIterations = parametersStructure.numberOfIterations;
    if ~IsNaturalNumber(numberOfIterations)
        error('numberOfIterations must be a natural number');
    end
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% First perform strip analysis on the coarseRefFrame. 
if numberOfIterations > 0
    [~, usefulEyePositionTraces, timeArray, ~] = ...
        StripAnalysis(inputVideoPath, coarseRefFrame, parametersStructure);

    if logical(abortTriggered)
        refinedFrame = [];
        return;
    end
else
    newRefFrame = coarseRefFrame;
end

%% For a certain number of iterations specified by the user, pingpong back
% and forth between extracting positions and generating reference frames
% based on those positions
k = 0;
while k < numberOfIterations
    parametersStructure.positions = usefulEyePositionTraces;
    parametersStructure.time = timeArray;
    
    newRefFrame = MakeMontage(parametersStructure, inputVideoPath);

    % If this is not the last iteration, perform strip analysis using the new
    % reference frame. If this is the last iteration, do not execute this
    % suite because the reference frame has already been updated to its
    % final form.
    if k ~= numberOfIterations - 1
        [~, usefulEyePositionTraces, timeArray, ~] = ...
            StripAnalysis(inputVideoPath, newRefFrame, parametersStructure);
        
        if logical(abortTriggered)
            refinedFrame = [];
            return;
        end
    end
    
    k = k + 1;
end

% Replace remaining black regions with random noise
indices = newRefFrame == 0;
newRefFrame(indices) = mean(newRefFrame(~indices)) + (std(newRefFrame(~indices)) ...
    * randn(sum(sum(indices)), 1));

refinedFrame = newRefFrame;

end