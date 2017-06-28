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

% Inspect each saccade
saccadesIndices = find(saccades);
saccadesDiffs = diff(saccadesIndices);
i = 1;
saccadeStructs = [];
while i < size(saccadesIndices,2)
    
    % Find the end of this saccade
    j = i;
    while j <= size(saccadesDiffs,2) && saccadesDiffs(j) == 1
        j = j + 1;
    end
    
    startOfSaccade = saccadesIndices(i);
    endOfSaccade = saccadesIndices(j);
    
    % Store information about this saccade in a new struct.
    if isempty(saccadeStructs)
        saccadeStructs = [saccadeStructs struct()];
    else
        % Must copy an existing struct since concatenated structs in an
        % array must have the same set of fields. This copied struct will
        % have its fields overwritten with correct values below.
        saccadeStructs = [saccadeStructs saccadeStructs(1)];
    end
    
    % Onset Time
    % time stamp of the start of event
    saccadeStructs(end).onsetTime = timeArray(startOfSaccade);
    
    % Offset Time
    % time stamp of end of event
    saccadeStructs(end).offsetTime = timeArray(endOfSaccade);
    
    % x Start
    % x position at start of event
    saccadeStructs(end).xStart = eyePositionTraces(startOfSaccade,1);
    
    % x End
    % x position at end of event
    saccadeStructs(end).xEnd = eyePositionTraces(endOfSaccade,1);

    % y Start
    % y position at start of event
    saccadeStructs(end).yStart = eyePositionTraces(startOfSaccade,2);

    % y End
    % y position at end of event
    saccadeStructs(end).yEnd = eyePositionTraces(endOfSaccade,2);
    
    % Duration
    % (Offset Time) - (Onset Time)
    saccadeStructs(end).duration = ...
        saccadeStructs(end).offsetTime - saccadeStructs(end).onsetTime;
    
    % Amplitude x
    % abs( (x Start) - (x End) )
    saccadeStructs(end).amplitude.x = ...
        abs(saccadeStructs(end).xStart - saccadeStructs(end).xEnd);

    % Amplitude y
    % abs( (y Start) - (y End) )
    saccadeStructs(end).amplitude.y = ...
        abs(saccadeStructs(end).yStart - saccadeStructs(end).yEnd);
    
    % Amplitude Vector
    % pythagorean theorem applied to Amplitudes x and y
    saccadeStructs(end).amplitude.vector = ...
        sqrt(saccadeStructs(end).amplitude.x^2 + saccadeStructs(end).amplitude.y^2);
    
    % Amplitude Direction
    % apply |atand2d| to delta y / delta x
    % (gives range [-180, 180] degrees)
    saccadeStructs(end).amplitude.direction = ...
        atan2d(saccadeStructs(end).yEnd - saccadeStructs(end).yStart, ...
        saccadeStructs(end).xEnd - saccadeStructs(end).xStart);
    
    % x Positions
    % excerpt of eye traces containing x positions for this saccade
    saccadeStructs(end).position.x = ...
        eyePositionTraces(startOfSaccade:endOfSaccade,1);
    
    % y Positions
    % excerpt of eye traces containing y positions for this saccade
    saccadeStructs(end).position.y = ...
        eyePositionTraces(startOfSaccade:endOfSaccade,2);
    
    % time
    % excerpt of time array containing times for this saccade
    saccadeStructs(end).time = timeArray(startOfSaccade:endOfSaccade);

    % x Velocity
    % excerpt of x velocities for this saccade
    saccadeStructs(end).velocity.x = ...
        verticalVelocityDiffs(startOfSaccade:endOfSaccade);
    
    % y Velocity
    % excerpt of y velocities for this saccade
    saccadeStructs(end).velocity.y = ...
        horizontalVelocityDiffs(startOfSaccade:endOfSaccade);
    
    % Mean x Velocity
    % mean of x velocities for this saccade
    saccadeStructs(end).meanVelocity.x = ...
        mean(saccadeStructs(end).velocity.x);
    
    % Mean y Velocity
    % mean of y velocities for this saccade
    saccadeStructs(end).meanVelocity.y = ...
        mean(saccadeStructs(end).velocity.y);
    
    % Peak x Velocity
    % highest x velocity for this saccade
    saccadeStructs(end).peakVelocity.x = ...
        max(saccadeStructs(end).velocity.x);
    
    % Peak y Velocity
    % highest y velocity for this saccade
    saccadeStructs(end).peakVelocity.y = ...
        max(saccadeStructs(end).velocity.y);
    
    % x Acceleration
    % excerpt of x accelerations for this saccade
    saccadeStructs(end).acceleration.x = ...
        verticalAccelerationDiffs(startOfSaccade:endOfSaccade);
    
    % y Acceleration
    % excerpt of y accelerations for this saccade
    saccadeStructs(end).acceleration.y = ...
        horizontalAccelerationDiffs(startOfSaccade:endOfSaccade);
    
    % Continue to next saccade
    i = j + 1;
end

% Inspect each drift
% Anything that was not a saccade is a drift
drifts = ~saccades;
driftsIndices = find(drifts);
driftsDiffs = diff(driftsIndices);
i = 1;
driftStructs = [];

while i < size(driftsIndices,2)
    
    % Find the end of this drift
    j = i;
    while j <= size(driftsDiffs,2) && driftsDiffs(j) == 1
        j = j + 1;
    end
    
    startOfdrift = driftsIndices(i);
    endOfdrift = driftsIndices(j);
    
    % Store information about this drift in a new struct.
    if isempty(driftStructs)
        driftStructs = [driftStructs struct()];
    else
        % Must copy an existing struct since concatenated structs in an
        % array must have the same set of fields. This copied struct will
        % have its fields overwritten with correct values below.
        driftStructs = [driftStructs driftStructs(1)];
    end
    
    % Onset Time
    % time stamp of the start of event
    driftStructs(end).onsetTime = timeArray(startOfdrift);
    
    % Offset Time
    % time stamp of end of event
    driftStructs(end).offsetTime = timeArray(endOfdrift);
    
    % x Start
    % x position at start of event
    driftStructs(end).xStart = eyePositionTraces(startOfdrift,1);
    
    % x End
    % x position at end of event
    driftStructs(end).xEnd = eyePositionTraces(endOfdrift,1);

    % y Start
    % y position at start of event
    driftStructs(end).yStart = eyePositionTraces(startOfdrift,2);

    % y End
    % y position at end of event
    driftStructs(end).yEnd = eyePositionTraces(endOfdrift,2);
    
    % Duration
    % (Offset Time) - (Onset Time)
    driftStructs(end).duration = ...
        driftStructs(end).offsetTime - driftStructs(end).onsetTime;
    
    % Amplitude x
    % abs( (x Start) - (x End) )
    driftStructs(end).amplitude.x = ...
        abs(driftStructs(end).xStart - driftStructs(end).xEnd);

    % Amplitude y
    % abs( (y Start) - (y End) )
    driftStructs(end).amplitude.y = ...
        abs(driftStructs(end).yStart - driftStructs(end).yEnd);
    
    % Amplitude Vector
    % pythagorean theorem applied to Amplitudes x and y
    driftStructs(end).amplitude.vector = ...
        sqrt(driftStructs(end).amplitude.x^2 + driftStructs(end).amplitude.y^2);
    
    % Amplitude Direction
    % apply |atand2d| to delta y / delta x
    % (gives range [-180, 180] degrees)
    driftStructs(end).amplitude.direction = ...
        atan2d(driftStructs(end).yEnd - driftStructs(end).yStart, ...
        driftStructs(end).xEnd - driftStructs(end).xStart);
    
    % x Positions
    % excerpt of eye traces containing x positions for this drift
    driftStructs(end).position.x = ...
        eyePositionTraces(startOfdrift:endOfdrift,1);
    
    % y Positions
    % excerpt of eye traces containing y positions for this drift
    driftStructs(end).position.y = ...
        eyePositionTraces(startOfdrift:endOfdrift,2);
    
    % time
    % excerpt of time array containing times for this drift
    driftStructs(end).time = timeArray(startOfdrift:endOfdrift);

    % x Velocity
    % excerpt of x velocities for this drift
    driftStructs(end).velocity.x = ...
        verticalVelocityDiffs(startOfdrift:endOfdrift);
    
    % y Velocity
    % excerpt of y velocities for this drift
    driftStructs(end).velocity.y = ...
        horizontalVelocityDiffs(startOfdrift:endOfdrift);
    
    % Mean x Velocity
    % mean of x velocities for this drift
    driftStructs(end).meanVelocity.x = ...
        mean(driftStructs(end).velocity.x);
    
    % Mean y Velocity
    % mean of y velocities for this drift
    driftStructs(end).meanVelocity.y = ...
        mean(driftStructs(end).velocity.y);
    
    % Peak x Velocity
    % highest x velocity for this drift
    driftStructs(end).peakVelocity.x = ...
        max(driftStructs(end).velocity.x);
    
    % Peak y Velocity
    % highest y velocity for this drift
    driftStructs(end).peakVelocity.y = ...
        max(driftStructs(end).velocity.y);
    
    % x Acceleration
    % excerpt of x accelerations for this drift
    driftStructs(end).acceleration.x = ...
        verticalAccelerationDiffs(startOfdrift:endOfdrift);
    
    % y Acceleration
    % excerpt of y accelerations for this drift
    driftStructs(end).acceleration.y = ...
        horizontalAccelerationDiffs(startOfdrift:endOfdrift);
    
    % Continue to next drift
    i = j + 1;
end

%% Save to output mat file
save(outputFileName, 'saccadeStructs', 'driftStructs');

end

