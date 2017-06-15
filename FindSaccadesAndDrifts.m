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

degreesPerPixelVertical = ...
    originalVideoSizeDegrees(1) / originalVideoSizePixels(1);
degreesPerPixelHorizontal = ...
    originalVideoSizeDegrees(2) / originalVideoSizePixels(2);

%eyePositionTraces(:,1) = eyePositionTraces(:,1) * degreesPerPixelVertical; %#ok<NODEF>
%eyePositionTraces(:,2) = eyePositionTraces(:,2) * degreesPerPixelHorizontal;

% Verbosity for debugging purposes
close all;
figure;
plot(timeArray, eyePositionTraces(:,1), ...
    timeArray, eyePositionTraces(:,2)); %#ok<NODEF>
title('Eye Position Traces');

%% Saccade detection algorithm

% 2 methods to calculate velocity
% (as selected by |inputParametersStructure.velocityMethod|)
%   - Method 1: v_x(t) = [x(t + \Delta t) - x(t)] / [\Delta t]
%   - Method 2: v_x(t) = [x(t + \Delta t) - x(t - \Delta t)] / [2 \Delta t]
if velocityMethod == 1
    verticalVelocityDiffs = diff(eyePositionTraces(:,1)) ./ diff(timeArray);
    horizontalVelocityDiffs = diff(eyePositionTraces(:,2)) ./ diff(timeArray);
elseif velocityMethod == 2
    verticalDiffs = nan(size(eyePositionTraces,1)-2,1);
    horizontalDiffs = nan(size(eyePositionTraces,1)-2,1);
    timeDiffs = nan(size(eyePositionTraces,1)-2,1);
    
    for i = 1:size(eyePositionTraces,1)-2
        verticalDiffs(i) = ...
            eyePositionTraces(i+2,1) - eyePositionTraces(i,1);
        horizontalDiffs(i) = ...
            eyePositionTraces(i+2,2) - eyePositionTraces(i,2);
        timeDiffs(i) = ...
            timeArray(i+2) - timeArray(i);
    end
    
    verticalVelocityDiffs = verticalDiffs ./ timeDiffs;
    horizontalVelocityDiffs = horizontalDiffs ./ timeDiffs;
    
    % Erasing temporary variable to avoid confusion
    clear verticalDiffs;
    clear horizontalDiffs;
else
    error('|inputParametersStructure.velocityMethod| must be 1 or 2');
end

% Use the differences in the vertical velocity to identify saccades
medianOfVerticalVelocityDiffs = median(verticalVelocityDiffs);
standardDeviationOfVerticalVelocityDiffs = ...
    sqrt(median(verticalVelocityDiffs.^2) - medianOfVerticalVelocityDiffs^2);
verticalVelocityThresholdLower = ...
    medianOfVerticalVelocityDiffs - thresholdValue * standardDeviationOfVerticalVelocityDiffs;
verticalVelocityThresholdUpper = ...
    medianOfVerticalVelocityDiffs + thresholdValue * standardDeviationOfVerticalVelocityDiffs;
if velocityMethod == 1
    verticalVelocityDiffs = [0; verticalVelocityDiffs];
elseif velocityMethod == 2
    verticalVelocityDiffs = [0; verticalVelocityDiffs; 0];
end
verticalSaccades = or(verticalVelocityDiffs' < verticalVelocityThresholdLower, ...
    verticalVelocityDiffs' > verticalVelocityThresholdUpper);

% Use the differences in the horizontal velocity to identify saccades
medianOfHorizontalVelocityDiffs = median(horizontalVelocityDiffs);
standardDeviationOfHorizontalVelocityDiffs = ...
    sqrt(median(horizontalVelocityDiffs.^2) - medianOfHorizontalVelocityDiffs^2);
horizontalVelocityThresholdLower = ...
    medianOfHorizontalVelocityDiffs - thresholdValue * standardDeviationOfHorizontalVelocityDiffs;
horizontalVelocityThresholdUpper = ...
    medianOfHorizontalVelocityDiffs + thresholdValue * standardDeviationOfHorizontalVelocityDiffs;
if velocityMethod == 1
    horizontalVelocityDiffs = [0; horizontalVelocityDiffs];
elseif velocityMethod == 2
    horizontalVelocityDiffs = [0; horizontalVelocityDiffs; 0];
end
horizontalSaccades = or(horizontalVelocityDiffs' < horizontalVelocityThresholdLower, ...
    horizontalVelocityDiffs' > horizontalVelocityThresholdUpper);

% Verbosity for debugging purposes
figure;
plot(1:size(verticalVelocityDiffs,1), verticalVelocityDiffs, ...
    1:size(verticalVelocityDiffs,1), ones(size(verticalVelocityDiffs,1),1)*verticalVelocityThresholdLower, ...
    1:size(verticalVelocityDiffs,1), ones(size(verticalVelocityDiffs,1),1)*verticalVelocityThresholdUpper)
title('Vertical Velocity Diffs');

figure;
plot(1:size(horizontalVelocityDiffs,1), horizontalVelocityDiffs, ...
    1:size(horizontalVelocityDiffs,1), ones(size(horizontalVelocityDiffs,1),1)*horizontalVelocityThresholdLower, ...
    1:size(horizontalVelocityDiffs,1), ones(size(horizontalVelocityDiffs,1),1)*horizontalVelocityThresholdUpper)
title('Horizontal Velocity Diffs');

% Now use the secondary velocity thresholds to capture entire peak of those
% identified with the first velocity thresholds.

% Doing this for verticalVelocitySaccades.
secondaryVerticalVelocityThresholdLower = ...
    medianOfVerticalVelocityDiffs - secondaryThresholdValue * standardDeviationOfVerticalVelocityDiffs;
secondaryVerticalVelocityThresholdUpper = ...
    medianOfVerticalVelocityDiffs + secondaryThresholdValue * standardDeviationOfVerticalVelocityDiffs;
i = 1;
while i <= size(verticalSaccades, 2)
    if ~verticalSaccades(i) % this time is not part of a saccade
        if i > 1 && verticalSaccades(i-1) && ...
                (verticalVelocityDiffs(i) < secondaryVerticalVelocityThresholdLower || ...
                verticalVelocityDiffs(i) > secondaryVerticalVelocityThresholdUpper)
            % Mark this time if the time before this one is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, and if this time should be identified as
            % a saccade since it meets one of the threshold requirements.
            verticalSaccades(i) = 1;
            % We now continue on since we already checked that the previous
            % time is marked.
            i = i + 1;
        else
            % Otherwise we continue without marking this time.
            i = i + 1;
        end
    else % this time is part of a saccade
        if i > 1 && ~verticalSaccades(i-1) && ...
                (verticalVelocityDiffs(i-1) < secondaryVerticalVelocityThresholdLower || ...
                verticalVelocityDiffs(i-1) > secondaryVerticalVelocityThresholdUpper)
            % Mark the time before this one if this time is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, if the time before has not been identified as a
            % saccade yet, and if the time before should be identified as
            % a saccade since it meets one of the threshold requirements.
            verticalSaccades(i-1) = 1;
            % We must now check the time before the one we just marked
            % too.
            i = i - 1;
        else
            i = i + 1;
        end
    end
end
% Repeat same logic but for horizonalSaccades
secondaryHorizontalThresholdLower = ...
    medianOfHorizontalVelocityDiffs - secondaryThresholdValue * standardDeviationOfHorizontalVelocityDiffs;
secondaryHorizontalThresholdUpper = ...
    medianOfHorizontalVelocityDiffs + secondaryThresholdValue * standardDeviationOfHorizontalVelocityDiffs;
i = 1;
while i <= size(horizontalSaccades, 2)
    if ~horizontalSaccades(i) % this time is not part of a saccade
        if i > 1 && horizontalSaccades(i-1) && ...
                (horizontalVelocityDiffs(i) < secondaryHorizontalThresholdLower || ...
                horizontalVelocityDiffs(i) > secondaryHorizontalThresholdUpper)
            % Mark this time if the time before this one is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, and if this time should be identified as
            % a saccade since it meets one of the threshold requirements.
            horizontalSaccades(i) = 1;
            % We now continue on since we already checked that the previous
            % time is marked.
            i = i + 1;
        else
            % Otherwise we continue without marking this time.
            i = i + 1;
        end
    else % this time is part of a saccade
        if i > 1 && ~horizontalSaccades(i-1) && ...
                (horizontalVelocityDiffs(i-1) < secondaryHorizontalThresholdLower || ...
                horizontalVelocityDiffs(i-1) > secondaryHorizontalThresholdUpper)
            % Mark the time before this one if this time is part of a
            % saccade, if we won't cause an index out of bounds error by
            % doing so, if the time before has not been identified as a
            % saccade yet, and if the time before should be identified as
            % a saccade since it meets one of the threshold requirements.
            horizontalSaccades(i-1) = 1;
            % We must now check the time before the one we just marked
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
saccades = or(verticalSaccades, horizontalSaccades);

%% Lump together microsaccades that are < |stitchCriteria| ms apart

% If the difference between any two marked saccades is less than
% |stitchCriteria|, then lump them together as one.
saccadesIndices = find(saccades);
for i = diff(saccadesIndices)
    if i > 1 && i < stitchCriteria
        for j = 1:i
            saccades(saccadesIndices(i)+j) = 1;
        end
    end
end

%% Remove saccades that have amplitude < |minAmplitude| or are > |maxDuration| in length

saccadesIndices = find(saccades);
saccadesDiffs = diff(saccadesIndices);

% Inspect each saccade
i = 1;
while i < size(saccadesIndices,2)
    
    % Find the end of this saccade
    j = i;
    while j <= size(saccadesDiffs,2) && saccadesDiffs(j) == 1
        j = j + 1;
    end
    
    startOfSaccade = saccadesIndices(i);
    endOfSaccade = saccadesIndices(j);
    
    % Check if this saccade is > |maxDuration| in length
    startTime = timeArray(startOfSaccade);
    endTime = timeArray(endOfSaccade);
    
    if endTime - startTime > maxDuration/1000
        % If it is, then remove this saccade
        for k = startOfSaccade:endOfSaccade
           saccades(k) = 0; 
        end
        
        % Continue to next saccade
        i = j + 1;
        continue;
    end
    
    % Check if this saccade has amplitude < |minAmplitude|
    amplitude = ...
        sqrt((eyePositionTraces(startOfSaccade,1) - eyePositionTraces(endOfSaccade,1))^2 + ...
        (eyePositionTraces(startOfSaccade,2) - eyePositionTraces(endOfSaccade,2))^2);
    
    if amplitude < minAmplitude
        % If it is, then remove this saccade
        for k = startOfSaccade:endOfSaccade
           saccades(k) = 0; 
        end
    end
    
    % Continue to next saccade in any case
    i = j + 1;
    continue;
end

%% Calculate acclerations
% (we use the same method that we used for velocity to get second
% derivative)
if velocityMethod == 1
    verticalAccelerationDiffs = diff(verticalVelocityDiffs) ./ diff(timeArray);
    horizontalAccelerationDiffs = diff(horizontalVelocityDiffs) ./ diff(timeArray);

    verticalAccelerationDiffs = [0; verticalAccelerationDiffs];
    horizontalAccelerationDiffs = [0; horizontalAccelerationDiffs];
elseif velocityMethod == 2
    verticalDiffs = nan(size(verticalVelocityDiffs,1)-2,1);
    horizontalDiffs = nan(size(horizontalVelocityDiffs,1)-2,1);
    % timeDiffs are the same as before, do not need to recalculate

    for i = 1:size(verticalVelocityDiffs,1)-2
        % note that: size(verticalDiffs) == size(horizontalDiffs)
        verticalDiffs(i) = ...
            verticalVelocityDiffs(i+2) - verticalVelocityDiffs(i);
        horizontalDiffs(i) = ...
            horizontalVelocityDiffs(i+2) - horizontalVelocityDiffs(i);
    end

    verticalAccelerationDiffs = verticalDiffs ./ timeDiffs;
    horizontalAccelerationDiffs = horizontalDiffs ./ timeDiffs;

    verticalAccelerationDiffs = [0; verticalAccelerationDiffs; 0];
    horizontalAccelerationDiffs = [0; horizontalAccelerationDiffs; 0];

    % Erasing temporary variables to avoid confusion
    clear verticalDiffs;
    clear horizontalDiffs;
end

%% Store results as an array of saccade structs and drift structs

% This flag is used to control the flow of the loop below. We first want to
% run the loop on all saccades. Once that is finished, this flag will help
% reset the loop and prepare the loop for execution on all the drifts.
% Finally, once that is done, the flag will help this loop to terminate.
loopFlag = false;

% The loop below will execute two rounds, once to handle all analysis of
% saccades, and then again for all drifts. For readability, the code below
% contains variables/comments refering to saccades. |loopFlag| helps us track
% whether we are processing saccades or drifts; if it is false, we are
% processing saccades; if it is true, we are processing drifts. Once round
% one (processing saccades) is completed, only then will we enter the if
% statement at the top of the loop. There, we will raise the flag, reset
% the loop, switch to drifts by inverting the logical array |saccades|
% which tells us which times contain eye traces that were part of a saccade
% (since anything that isn't part of a saccade is a drift), and transfer
% out the data of and reset |arrayOfStructs|. Once this is complete, then
% round two of the loop has begun for the analysis of drifts. The loop
% terminates once both rounds are complete since |i| will reach the end of
% its matrix size when drifts are complete and also |loopFlag| will have
% been raised since saccades are complete.

% Inspect each saccade
saccadesIndices = find(saccades);
saccadesDiffs = diff(saccadesIndices);
i = 1;
arrayOfStructs = [];
while i < size(saccadesIndices,2) || ~loopFlag
    
    % This if statement entered if we are done with saccades.
    % In here, we prepare the loop to run again for drifts.
    if i >= size(saccadesIndices,2)
        % Raise the flag to indicate that we are starting drifts.
        loopFlag = true;
        
        % Reset loop variable
        i = 1;
        
        % Anything that was not a saccade is a drift
        saccades = ~saccades;
        saccadesIndices = find(saccades);
        saccadesDiffs = diff(saccadesIndices);
        
        % Transfer and clear |arrayOfStructs|
        % It will now store drift structs rather than saccade structs
        saccadeStructs = arrayOfStructs;
        arrayOfStructs = [];
        
        % Check the loop condition again before proceeding.
        % This prevents index out of bounds below if there are no drifts.
        continue;
    end
    
    % Find the end of this saccade
    j = i;
    while j <= size(saccadesDiffs,2) && saccadesDiffs(j) == 1
        j = j + 1;
    end
    
    startOfSaccade = saccadesIndices(i);
    endOfSaccade = saccadesIndices(j);
    
    % Store information about this saccade in a new struct.
    if isempty(arrayOfStructs)
        arrayOfStructs = [arrayOfStructs struct()];
    else
        % Must copy an existing struct since concatenated structs in an
        % array must have the same set of fields. This copied struct will
        % have its fields overwritten with correct values below.
        arrayOfStructs = [arrayOfStructs arrayOfStructs(1)];
    end
    
    % Onset Time
    % time stamp of the start of event
    arrayOfStructs(end).onsetTime = timeArray(startOfSaccade);
    
    % Offset Time
    % time stamp of end of event
    arrayOfStructs(end).offsetTime = timeArray(endOfSaccade);
    
    % x Start
    % x position at start of event
    arrayOfStructs(end).xStart = eyePositionTraces(startOfSaccade,1);
    
    % x End
    % x position at end of event
    arrayOfStructs(end).xEnd = eyePositionTraces(endOfSaccade,1);

    % y Start
    % y position at start of event
    arrayOfStructs(end).yStart = eyePositionTraces(startOfSaccade,2);

    % y End
    % y position at end of event
    arrayOfStructs(end).yEnd = eyePositionTraces(endOfSaccade,2);
    
    % Duration
    % (Offset Time) - (Onset Time)
    arrayOfStructs(end).duration = ...
        arrayOfStructs(end).offsetTime - arrayOfStructs(end).onsetTime;
    
    % Amplitude x
    % abs( (x Start) - (x End) )
    arrayOfStructs(end).amplitude.x = ...
        abs(arrayOfStructs(end).xStart - arrayOfStructs(end).xEnd);

    % Amplitude y
    % abs( (y Start) - (y End) )
    arrayOfStructs(end).amplitude.y = ...
        abs(arrayOfStructs(end).yStart - arrayOfStructs(end).yEnd);
    
    % Amplitude Vector
    % pythagorean theorem applied to Amplitudes x and y
    arrayOfStructs(end).amplitude.vector = ...
        sqrt(arrayOfStructs(end).amplitude.x^2 + arrayOfStructs(end).amplitude.y^2);
    
    % Amplitude Direction
    % apply |atand2d| to delta y / delta x
    % (gives range [-180, 180] degrees)
    arrayOfStructs(end).amplitude.direction = ...
        atan2d(arrayOfStructs(end).yEnd - arrayOfStructs(end).yStart, ...
        arrayOfStructs(end).xEnd - arrayOfStructs(end).xStart);
    
    % x Positions
    % excerpt of eye traces containing x positions for this saccade
    arrayOfStructs(end).position.x = ...
        eyePositionTraces(startOfSaccade:endOfSaccade,1);
    
    % y Positions
    % excerpt of eye traces containing y positions for this saccade
    arrayOfStructs(end).position.y = ...
        eyePositionTraces(startOfSaccade:endOfSaccade,2);
    
    % time
    % excerpt of time array containing times for this saccade
    arrayOfStructs(end).time = timeArray(startOfSaccade:endOfSaccade);

    % x Velocity
    % excerpt of x velocities for this saccade
    arrayOfStructs(end).velocity.x = ...
        verticalVelocityDiffs(startOfSaccade:endOfSaccade);
    
    % y Velocity
    % excerpt of y velocities for this saccade
    arrayOfStructs(end).velocity.y = ...
        horizontalVelocityDiffs(startOfSaccade:endOfSaccade);
    
    % Mean x Velocity
    % mean of x velocities for this saccade
    arrayOfStructs(end).meanVelocity.x = ...
        mean(arrayOfStructs(end).velocity.x);
    
    % Mean y Velocity
    % mean of y velocities for this saccade
    arrayOfStructs(end).meanVelocity.y = ...
        mean(arrayOfStructs(end).velocity.y);
    
    % Peak x Velocity
    % highest x velocity for this saccade
    arrayOfStructs(end).peakVelocity.x = ...
        max(arrayOfStructs(end).velocity.x);
    
    % Peak y Velocity
    % highest y velocity for this saccade
    arrayOfStructs(end).peakVelocity.y = ...
        max(arrayOfStructs(end).velocity.y);
    
    % x Acceleration
    % excerpt of x accelerations for this saccade
    arrayOfStructs(end).acceleration.x = ...
        verticalAccelerationDiffs(startOfSaccade:endOfSaccade);
    
    % y Acceleration
    % excerpt of y accelerations for this saccade
    arrayOfStructs(end).acceleration.y = ...
        horizontalAccelerationDiffs(startOfSaccade:endOfSaccade);
    
    % Continue to next saccade
    i = j + 1;
end

driftStructs = arrayOfStructs;

%% Save to output mat file
save(outputFileName, 'saccadeStructs', 'driftStructs');

end

