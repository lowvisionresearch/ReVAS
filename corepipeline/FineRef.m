function [newRefFrame, usefulEyePositionTraces, timeArray] = ...
    FineRef(coarseRefFrame, inputVideo, parametersStructure)
%FINE REF  Generate a better reference frame.
%   The function alternates between StripAnalysis and MakeMontage,
%   alternating between generating eye traces and generating the reference
%   frames that result from those traces.
%   
%   -----------------------------------
%   Input
%   -----------------------------------
%   |coarseRefFrame| is the path to the coarse reference frame or a matrix 
%   representation of the coarse reference frame
%
%   |inputVideo| is the path to the video or the video matrix.
%
%   |parametersStructure| is a struct as specified below.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  numberOfIterations  :   number of strip-analysis-to-reference-frame
%                          cycles to perform. For example, when set to 1, one
%                          StripAnalysis will be performed on the
%                          coarseRefFrame, and a fine reference frame will 
%                          be generated from those eye position traces. If 
%                          set to 2, another strip analysis will be
%                          performed on that fine reference frame, and
%                          another fine reference frame will be generated
%                          from the resulting eye position traces
%                          (default 1)
%   
%   Note: FineRef also calls StripAnalysis and MakeMontage. Refer to those
%   functions for additional parameters.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       coarseRefFrame = load('MyVid_coarseRef.mat');
%       parametersStructure = load(MyVid_params.mat');
%       newRefFrame = FineRef(coarseRefFrame, inputVideo, parametersStructure);

%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
end

%% Handle overwrite scenarios.
if writeResult
    outputFilePath = Filename(inputVideo, 'fineref');

    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasWarning(['FineRef() did not execute because it would overwrite existing file. (' outputFilePath ')'], parametersStructure);
        data = load(outputFilePath);
        newRefFrame = data.refFrame;
        usefulEyePositionTraces = data.eyePositionTraces;
        timeArray = data.timeArray;
        return;
    else
        RevasWarning(['FineRef() is proceeding and overwriting an existing file. (' outputFilePath ')'], parametersStructure);  
    end
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'numberOfIterations')
    parametersStructure.numberOfIterations = 1;
else
    if ~IsNaturalNumber(parametersStructure.numberOfIterations)
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
if parametersStructure.numberOfIterations > 0
    [~, usefulEyePositionTraces, timeArray, ~] = ...
        StripAnalysis(inputVideo, coarseRefFrame, parametersStructure);

    if logical(abortTriggered)
        newRefFrame = [];
        return;
    end
else
    % If user specifies 0 iterations, set the return value to the coarse
    % reference frame that was passed in.
    newRefFrame = coarseRefFrame;
    if writeResult
        outputFilePath = inputVideo;
        outputFilePath(end-3:end) = [];
        outputFilePath(end+1:end+9) = '_refframe';
        save(outputFilePath, 'newRefFrame');
    end
end

%% For a certain number of iterations specified by the user, pingpong back
% and forth between extracting positions and generating reference frames
% based on those positions
k = 0;
while k < parametersStructure.numberOfIterations
    parametersStructure.positions = usefulEyePositionTraces;
    parametersStructure.time = timeArray;
    
    % If this is the final iteration, add a flag to the parametersStructure
    % to add random noise to black regions of the final reference frame.
    if k + 1 == parametersStructure.numberOfIterations
        parametersStructure.addNoise = true;
    end
    
    newRefFrame = MakeMontage(parametersStructure, inputVideo);
    % If this is not the last iteration, perform strip analysis using the new
    % reference frame. If this is the last iteration, do not execute this
    % suite because the reference frame has already been updated to its
    % final form.
    if k ~= parametersStructure.numberOfIterations - 1
        [~, usefulEyePositionTraces, timeArray, ~] = ...
            StripAnalysis(inputVideo, newRefFrame, parametersStructure);
        
        if logical(abortTriggered)
            newRefFrame = [];
            return;
        end
    end
    
    k = k + 1;
end
end
