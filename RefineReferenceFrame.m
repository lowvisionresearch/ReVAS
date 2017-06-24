function refinedFrame = RefineReferenceFrame(coarseRefFrame, params)
%REFINE REFERENCE FRAME  Generate a better reference frame.
%   The function alternates between StripAnalysis and MakeMontage,
%   alternating between generating positions and generating the reference
%   frames that result from those positions
%   
%   params takes in the fields stripHeight, stripWidth, samplingRate,
%   enableSubpixelInterpolation,subpixelInterpolationParameters.neighborhoodSize, 
%   subpixelInterpolationParameters.subpixelDepth, adaptiveSearch,
%   adaptiveSearchScalingFactor,searchWindowHeight, badFrames, minimumPeakRatio,
%   minimumPeakThreshold, enableVerbosity, axesHandles, enableGPU,
%   videoPath, numberOfIterations.

% First perform strip analysis on the coarseRefFrame to get a rough
% estimate of the strip positions
if params.numberOfIterations > 0
    [~, usefulEyePositionTraces, timeArray, ~] = ...
        StripAnalysis(params.fileName, coarseRefFrame, params);
else
    newRefFrame = coarseRefFrame;
end

% For a certain number of iterations specified by the user, pingpong back
% and forth between extracting positions and generating reference frames
% based on those positions
k = 0;
while k < params.numberOfIterations
    params.positions = usefulEyePositionTraces;
    params.time = timeArray;
    
    newRefFrame = MakeMontage(params, params.fileName);
    
    % If this is not the last iteration, perform strip analysis using the new
    % reference frame. If this is the last iteration, do not execute this
    % suite because the reference frame has already been updated to its
    % final form.
    if k ~= params.numberOfIterations - 1
        [~, usefulEyePositionTraces, timeArray, ~] = ...
        StripAnalysis(params.fileName, newRefFrame, params);
    end
    
    k = k + 1;
end

refinedFrame = newRefFrame;

end