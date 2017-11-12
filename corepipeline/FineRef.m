function newRefFrame = FineRef(coarseRefFrame, inputVideoPath, parametersStructure)
%FINE REF  Generate a better reference frame.
%   The function alternates between StripAnalysis and MakeMontage,
%   alternating between generating eye traces and generating the reference
%   frames that result from those traces.
%   
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  numberOfIterations  :   number of strip-analysis-to-reference-frame
%                          cycles to perform (i.e., when set to 1, one
%                          StripAnalysis will be performed on the
%                          coarseRefFrame, and a fine reference frame will 
%                          be generated from those eye position traces. If 
%                          set to 2, another strip analysis will be
%                          performed on that fine reference frame, and
%                          another fine reference frame will be generated
%                          from the resulting eye position traces.)
%   
%   Note: FineRef also calls StripAnalysis and MakeMontage. Refer to those
%   functions for additional parameters.
%
%   Example usage: 
%       videoPath = 'MyVid.avi';
%       coarseRefFrame = load('MyVid_coarseRef.mat');
%       parametersStructure = load(MyVid_params.mat');
%       newRefFrame = FineRef(coarseRefFrame, videoPath, parametersStructure);

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
    
    % If this is the final iteration, add a flag to the parametersStructure
    % to add random noise to black regions of the final reference frame.
    if k + 1 == numberOfIterations
        parametersStructure.addNoise = true;
    end
    
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
end
