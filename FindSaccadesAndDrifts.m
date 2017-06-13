function FindSaccadesAndDrifts(inputEyePositionsFilePath, ...
    originalVideoSizePixels, originalVideoSizeDegrees, ...
    inputParametersStructure)
%FIND SACCADES AND DRIFTS Records in a mat file an array of structures
%representing saccades and an array of structures representing drifts.
%   The result is stored with '_sacsdrifts' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputFileName = [inputEyePositionsFilePath(1:end-4) '_sacsdrifts'];

%% Handle overwrite scenarios.
if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(inputParametersStructure, 'overwrite') || ~inputParametersStructure.overwrite
    warning('FindSaccadesAndDrifts() did not execute because it would overwrite existing file.');
    return;
else
    warning('FindSaccadesAndDrifts() is proceeding and overwriting an existing file.');
end

%% Set input variables to defaults if not provided

if ~isfield(inputParametersStructure, 'thresholdValue')
    thresholdValue = 6;
else
    thresholdValue = inputParametersStructure.thresholdValue;
end

if ~isfield(inputParametersStructure, 'secondaryThresholdValue')
    secondaryThresholdValue = 2;
else
    secondaryThresholdValue = inputParametersStructure.secondaryThresholdValue;
end

% units are in milliseconds
if ~isfield(inputParametersStructure, 'stitchCriteria')
    stitchCriteria = 15;
else
    stitchCriteria = inputParametersStructure.stitchCriteria;
end

% units are in degrees
if ~isfield(inputParametersStructure, 'minAmplitude')
    minAmplitude = 0.1;
else
    minAmplitude = inputParametersStructure.minAmplitude;
end

% units are in milliseconds
if ~isfield(inputParametersStructure, 'maxDuration')
    maxDuration = 100;
else
    maxDuration = inputParametersStructure.maxDuration;
end

% Method to use for detection
% 1 = Hard velocity threshold
% 2 = Median-based (default)
if ~isfield(inputParametersStructure, 'detectionMethod')
    detectionMethod = 2;
else
    detectionMethod = inputParametersStructure.detectionMethod;
end

% units are in degrees/second
if ~isfield(inputParametersStructure, 'hardVelocityThreshold')
    hardVelocityThreshold = 35;
else
    hardVelocityThreshold = inputParametersStructure.hardVelocityThreshold;
end

% Method to use to calculate velocity
% 1 = using |diff|
% 2 = (x_(n+1) - x_(n-1)) / 2 delta t)
if ~isfield(inputParametersStructure, 'velocityMethod')
    velocityMethod = 2;
else
    velocityMethod = inputParametersStructure.velocityMethod;
end

%% Load mat file with output from |StripAnalysis|

load(inputEyePositionsFilePath);

% Variables that should be Loaded now:
% - eyePositionTraces
% - parametersStructure
% - referenceFramePath
% - timeArray

%% Convert eye position traces from pixels to degrees

degreesPerPixelHorizontal = ...
    originalVideoSizeDegrees(2) / originalVideoSizePixels(2);
degreesPerPixelVertical = ...
    originalVideoSizeDegrees(1) / originalVideoSizePixels(1);

eyePositionTraces(:,1) = eyePositionTraces(:,1) * degreesPerPixelHorizontal;
eyePositionTraces(:,2) = eyePositionTraces(:,2) * degreesPerPixelVertical;

%% Saccade detection algorithm

% Use the differences in the vertical direction to identify saccade frames
verticalDiffs = diff(eyePositionTraces(:,1));
medianOfVerticalDiffs = median(verticalDiffs);
standardDeviationOfVerticalDiffs = sqrt(median(verticalDiffs.^2) - medianOfVerticalDiffs^2);
verticalThresholdLower = medianOfVerticalDiffs - thresholdValue * standardDeviationOfVerticalDiffs;
verticalThresholdUpper = medianOfVerticalDiffs + thresholdValue * standardDeviationOfVerticalDiffs;
verticalDiffs = [0; verticalDiffs];
verticalSaccadeFrames = or(verticalDiffs' < verticalThresholdLower, ...
    verticalDiffs' > verticalThresholdUpper);

% Use the differences in the horizontal direction to identify saccade frames
horizontalDiffs = diff(eyePositionTraces(:,2));
medianOfHorizontalDiffs = median(horizontalDiffs);
standardDeviationOfHorizontalDiffs = ...
    sqrt(median(horizontalDiffs.^2) - medianOfHorizontalDiffs^2);
horizontalThresholdLower = ...
    medianOfHorizontalDiffs - thresholdValue * standardDeviationOfHorizontalDiffs;
horizontalThresholdUpper = ...
    medianOfHorizontalDiffs + thresholdValue * standardDeviationOfHorizontalDiffs;
horizontalDiffs = [0; horizontalDiffs];
horizontalSaccadeFrames = or(horizontalDiffs' < horizontalThresholdLower, ...
    horizontalDiffs' > horizontalThresholdUpper);

% Verbosity for debugging purposes
%close all;
%figure;
%plot(1:size(verticalDiffs,1), verticalDiffs, ...
%    1:size(verticalDiffs,1), ones(size(verticalDiffs,1),1)*verticalThresholdLower, ...
%    1:size(verticalDiffs,1), ones(size(verticalDiffs,1),1)*verticalThresholdUpper)
%title('Vertical Diffs');

%figure;
%plot(1:size(horizontalDiffs,1), horizontalDiffs, ...
%    1:size(horizontalDiffs,1), ones(size(horizontalDiffs,1),1)*horizontalThresholdLower, ...
%    1:size(horizontalDiffs,1), ones(size(horizontalDiffs,1),1)*horizontalThresholdUpper)
%title('Horizontal Diffs');

% Now use the secondaryThresholdValue to capture entire peak of those
% identified with the first thresholdValue.

% Doing this for verticalSaccadeFrames.
secondaryVerticalThresholdLower = ...
    medianOfVerticalDiffs - secondaryThresholdValue * standardDeviationOfVerticalDiffs;
secondaryVerticalThresholdUpper = ...
    medianOfVerticalDiffs + secondaryThresholdValue * standardDeviationOfVerticalDiffs;
i = 1;
while i <= size(verticalSaccadeFrames, 2)
    if ~verticalSaccadeFrames(i) % this frame is not part of a saccade
        if i > 1 && verticalSaccadeFrames(i-1) && ...
                (verticalDiffs(i) < secondaryVerticalThresholdLower || ...
                verticalDiffs(i) > secondaryVerticalThresholdUpper)
            % Mark this frame if the frame before this one is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, and if this frame should be identified as
            % a saccade since it meets one of the threshold requirements.
            verticalSaccadeFrames(i) = 1;
            % We now continue on since we already checked that the previous
            % frame is marked.
            i = i + 1;
        else
            % Otherwise we continue without marking this frame.
            i = i + 1;
        end
    else % this frame is part of a saccade
        if i > 1 && ~verticalSaccadeFrames(i-1) && ...
                (verticalDiffs(i-1) < secondaryVerticalThresholdLower || ...
                verticalDiffs(i-1) > secondaryVerticalThresholdUpper)
            % Mark the frame before this one if this frame is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, if the frame before has not been identified as a
            % saccade yet, and if the frame before should be identified as
            % a saccade since it meets one of the threshold requirements.
            verticalSaccadeFrames(i-1) = 1;
            % We must now check the frame before the one we just marked
            % too.
            i = i - 1;
        else
            i = i + 1;
        end
    end
end
% Repeat same logic but for horizonalSaccadeFrames
secondaryHorizontalThresholdLower = ...
    medianOfHorizontalDiffs - secondaryThresholdValue * standardDeviationOfHorizontalDiffs;
secondaryHorizontalThresholdUpper = ...
    medianOfHorizontalDiffs + secondaryThresholdValue * standardDeviationOfHorizontalDiffs;
i = 1;
while i <= size(horizontalSaccadeFrames, 2)
    if ~horizontalSaccadeFrames(i) % this frame is not part of a saccade
        if i > 1 && horizontalSaccadeFrames(i-1) && ...
                (horizontalDiffs(i) < secondaryHorizontalThresholdLower || ...
                horizontalDiffs(i) > secondaryHorizontalThresholdUpper)
            % Mark this frame if the frame before this one is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, and if this frame should be identified as
            % a saccade since it meets one of the threshold requirements.
            horizontalSaccadeFrames(i) = 1;
            % We now continue on since we already checked that the previous
            % frame is marked.
            i = i + 1;
        else
            % Otherwise we continue without marking this frame.
            i = i + 1;
        end
    else % this frame is part of a saccade
        if i > 1 && ~horizontalSaccadeFrames(i-1) && ...
                (horizontalDiffs(i-1) < secondaryHorizontalThresholdLower || ...
                horizontalDiffs(i-1) > secondaryHorizontalThresholdUpper)
            % Mark the frame before this one if this frame is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, if the frame before has not been identified as a
            % saccade yet, and if the frame before should be identified as
            % a saccade since it meets one of the threshold requirements.
            horizontalSaccadeFrames(i-1) = 1;
            % We must now check the frame before the one we just marked
            % too.
            i = i - 1;
        else
            i = i + 1;
        end
    end
end

% Verbosity for debugging purposes
%figure;
%plot(1:size(verticalDiffs,1), verticalDiffs, ...
%    1:size(verticalDiffs,1), ones(size(verticalDiffs,1),1)*secondaryVerticalThresholdLower, ...
%    1:size(verticalDiffs,1), ones(size(verticalDiffs,1),1)*secondaryVerticalThresholdUpper)
%title('Vertical Diffs Secondary Threshold');

%figure;
%plot(1:size(horizontalDiffs,1), horizontalDiffs, ...
%    1:size(horizontalDiffs,1), ones(size(horizontalDiffs,1),1)*secondaryHorizontalThresholdLower, ...
%    1:size(horizontalDiffs,1), ones(size(horizontalDiffs,1),1)*secondaryHorizontalThresholdUpper)
%title('Horizontal Diffs Secondary Threshold');

% Combine results from both vertical and horizontal approaches
saccadeFrames = or(verticalSaccadeFrames, horizontalSaccadeFrames);

%% Save to output mat file
save(outputFileName, '');

end

