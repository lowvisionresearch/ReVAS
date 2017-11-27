function [outputFileName, saccades, drifts] = FindSaccadesAndDrifts(inputEyePositionsFilePath, ...
    originalVideoSizePixels, originalVideoSizeDegrees, ...
    parametersStructure)
%FIND SACCADES AND DRIFTS Records in a mat file an array of structures
%representing saccades and an array of structures representing drifts.
%   The result is stored with '_sacsdrifts' appended to the input video file
%   name.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  overwrite               : set to 1 to overwrite existing files resulting 
%                            from calling the function.
%                            Set to 0 to abort the function call if the
%                            files exist in the current directory.
%  thresholdValue          : multiplier specifying the median-based velocity
%                            threshold. The most typically used value is 6.
%  secondaryThresholdValue : multiplier specifying a secondary velocity
%                            threshold for more accurate detection of onset
%                            and offset of a saccade
%  stitchCriteria          : if two consecutive saccades are closer in time
%                            less than this value (in ms), then they are
%                            stitched back to back.
%  minAmplitude            : minimum (micro)saccade amplitude in deg
%  minDuration             : minimum saccade duration in ms
%  maxDuration             : maximum saccade duration in ms
%  detectionMethod         : saccade detection method. set to 1 for using a
%                            hard velocity threshold. set to 2 for using a
%                            median-based velocity threshold. defaults to 2.
%  velocityMethod          : set to 1 for regular differentiation. set to 2
%                            for three-point differentiation.
%  enableVerbosity         : set to 1 to see progress and/or output
%  axesHandles             : axis handles to plot the output. 
%  hardVelocityThreshold   : in deg/sec. relevant only when velocityMethod is
%                            set to 1.
%  hardSecondaryVelocityThreshold : in deg/sec. relevant only when
%                           velocityMethod is 1.
%
%
%
%
%% Handle overwrite scenarios.
outputFileName = [inputEyePositionsFilePath(1:end-4) '_sacsdrifts'];
if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['FindSaccadesAndDrifts() did not execute because it would overwrite existing file. (' outputFileName ')'], parametersStructure);
    return;
else
    RevasWarning(['FindSaccadesAndDrifts() is proceeding and overwriting an existing file. (' outputFileName ')'], parametersStructure);
end

%% Set parameters to defaults if not specified.
% This second method is the EK Algorithm (Engbert & Kliegl, 2003 Vision Research).
if ~isfield(parametersStructure, 'thresholdValue')
    lambda = 6;
    RevasWarning('using default parameter for lambda', parametersStructure);
else
    lambda = parametersStructure.thresholdValue;
end

if ~isfield(parametersStructure, 'secondaryThresholdValue')
    secondaryLambda = 3;
    RevasWarning('using default parameter for secondaryLambda', parametersStructure);
else
    secondaryLambda = parametersStructure.secondaryThresholdValue;
end

% units are in milliseconds
if ~isfield(parametersStructure, 'stitchCriteria')
    stitchCriteria = 15;
    RevasWarning('using default parameter for stitchCriteria', parametersStructure);
else
    stitchCriteria = parametersStructure.stitchCriteria;
end

% units are in degrees
if ~isfield(parametersStructure, 'minAmplitude')
    minAmplitude = 0.05;
    RevasWarning('using default parameter for minAmplitude', parametersStructure);
else
    minAmplitude = parametersStructure.minAmplitude;
end

% units are in milliseconds
if ~isfield(parametersStructure, 'minDuration')
    minDuration = 8;
    RevasWarning('using default parameter for minDuration', parametersStructure);
else
    minDuration = parametersStructure.minDuration;
end

% units are in milliseconds
if ~isfield(parametersStructure, 'maxDuration')
    maxDuration = 100;
    RevasWarning('using default parameter for maxDuration', parametersStructure);
else
    maxDuration = parametersStructure.maxDuration;
end

% Method to use for detection
% 1 = Hard velocity threshold
% 2 = Median-based (default)
if ~isfield(parametersStructure, 'detectionMethod')
    detectionMethod = 2;
else
    detectionMethod = parametersStructure.detectionMethod;
end

% units are in degrees/second
if ~isfield(parametersStructure, 'hardVelocityThreshold')
    hardVelocityThreshold = 25;
    if detectionMethod == 1
        RevasWarning('using default parameter for hardVelocityThreshold', parametersStructure);
    end
else
    hardVelocityThreshold = parametersStructure.hardVelocityThreshold;
end

% units are in degrees/second
if ~isfield(parametersStructure, 'hardSecondaryVelocityThreshold')
    hardSecondaryVelocityThreshold = 15;
    if detectionMethod == 1
        RevasWarning('using default parameter for hardSecondaryVelocityThreshold', parametersStructure);
    end
else
    hardSecondaryVelocityThreshold = parametersStructure.hardVelocityThreshold;
end

% Method to use to calculate velocity
% 1 = using |diff|
% 2 = (x_(n+1) - x_(n-1)) / 2 delta t)
if ~isfield(parametersStructure, 'velocityMethod')
    velocityMethod = 2;
else
    velocityMethod = parametersStructure.velocityMethod;
end

% check verbosity field
if ~isfield(parametersStructure, 'enableVerbosity')
    enableVerbosity = false;
else
    enableVerbosity = parametersStructure.enableVerbosity;
end

% save parameters structure in the environment before loading a different
% parameters structure
axesHandlesFlag = isfield(parametersStructure, 'axesHandles');
if axesHandlesFlag
    axesHandles = parametersStructure.axesHandles;
end

%% Load mat file with output from |StripAnalysis|
load(inputEyePositionsFilePath);
% Variables that should be loaded now:
% - eyePositionTraces
% - parametersStructure
% - referenceFramePath
% - timeArray


%% Convert eye position traces from pixels to degrees
degreesPerPixelVertical = ...
    originalVideoSizeDegrees(1) / originalVideoSizePixels(1);
degreesPerPixelHorizontal = ...
    originalVideoSizeDegrees(2) / originalVideoSizePixels(2);

eyePositionTraces(:,1) = eyePositionTraces(:,1) * degreesPerPixelVertical; 
eyePositionTraces(:,2) = eyePositionTraces(:,2) * degreesPerPixelHorizontal;

%% Saccade detection algorithm
% 2 methods to calculate velocity
% (as selected by |inputParametersStructure.velocityMethod|)
%   - Method 1: v_x(t) = [x(t + \Delta t) - x(t)] / [\Delta t]
%   - Method 2: v_x(t) = [x(t + \Delta t) - x(t - \Delta t)] / [2 \Delta t]
if velocityMethod == 1
    velocity = [0,0; diff(eyePositionTraces) ./ repmat(diff(timeArray),1,2)];
elseif velocityMethod == 2    
    len = size(timeArray,1);
    velocity = ...
        [0,0; (eyePositionTraces(3:len,:)-eyePositionTraces(1:len-2,:)) ./ ...
        repmat(timeArray(3:len)-timeArray(1:len-2),1,2); 0,0];
    
    % Erasing temporary variable to avoid confusion
    clear len;
else
    error('|inputParametersStructure.velocityMethod| must be 1 or 2');
end

%% Find saccades.
% Use the differences in the velocity to identify saccades
vectorialVelocity = sqrt(sum(velocity.^2,2));
medianVelocity = nanmedian(vectorialVelocity);
sdVelocity = sqrt(nanmedian(vectorialVelocity.^2) - medianVelocity.^2);
if detectionMethod == 1
    velocityThreshold = hardVelocityThreshold;
    secondaryVelocityThreshold = hardSecondaryVelocityThreshold;
else
    velocityThreshold = ...
        medianVelocity + lambda * sdVelocity;
    secondaryVelocityThreshold = medianVelocity + secondaryLambda * sdVelocity;
end

% it's enough to exceed the threshold in one dimension only
saccadeIndices = velocity > velocityThreshold;
saccadeIndices = saccadeIndices(:,1) | saccadeIndices(:,2);

% compute saccade onset and offset indices
[onsets, offsets] = GetEventOnsetsAndOffsets(saccadeIndices);

% remove artifacts
[onsets, offsets] = RemoveFakeSaccades(timeArray, ...
    onsets, offsets, stitchCriteria, minDuration, maxDuration, velocity);

% Get saccade properties
saccades = GetSaccadeProperties(eyePositionTraces,timeArray,onsets,offsets,velocity,...
    secondaryVelocityThreshold);

% filter out really small saccades
toRemove = [saccades.vectorAmplitude] < minAmplitude;
saccades(toRemove) = [];
onsets = [saccades.onsetIndex];
offsets = [saccades.offsetIndex];

%% Find drifts.
% Anything other than saccades and artifacts, will be drifts.
driftIndices = true(size(timeArray));
for i=1:length(onsets)
    driftIndices(onsets(i):offsets(i)) = false;
end

% try
%     % artifacts, the regions that cannot be classified as saccades but still
%     % exceeds the velocity thresholds.
%     beforeAfter = round(0.005/diff(timeArray(1:2))); 
%     remove = conv(double(saccadeIndices),ones(beforeAfter,1),'same')>0;
%     driftIndices(remove) = false;
% catch
%    % ignore 
% end

% compute drift onsets and offsets
[driftOnsets, driftOffsets] = GetDriftOnsetsAndOffsets(driftIndices);

% if the temporal gap between two consecutive drifts is less than minimum
% saccade duration, then stitch.
[driftOnsets, driftOffsets] = MergeDrifts(driftOnsets,driftOffsets,minDuration,timeArray);

% get drift parameters
drifts = GetDriftProperties(eyePositionTraces,timeArray,driftOnsets,driftOffsets,velocity); 


%% Save to output mat file.
save(outputFileName, 'saccades', 'drifts');

%% Verbosity for Results.
if enableVerbosity
    if axesHandlesFlag
        axes(axesHandles(2));
        colormap(axesHandles(2), 'default');
    else
        figure(1543);
    end
    cla;
    ax = gca;
    ax.ColorOrderIndex = 1;
    plot(timeArray, eyePositionTraces(:,1),'-'); hold on;
    plot(timeArray, eyePositionTraces(:,2),'-'); hold on;
    ax.ColorOrderIndex = 1;
    plot(timeArray(driftIndices), eyePositionTraces(driftIndices,1),'.','LineWidth',2); hold on;
    plot(timeArray(driftIndices), eyePositionTraces(driftIndices,2),'.','LineWidth',2); hold on;
    
    % now highlight saccades
    for i=1:length(onsets)
        ax.ColorOrderIndex = 1;
        plot(timeArray(onsets(i):offsets(i)),eyePositionTraces(onsets(i):offsets(i),1),'o',...
             timeArray(onsets(i):offsets(i)),eyePositionTraces(onsets(i):offsets(i),2),'o'); hold on;
    end
    title('Eye position');
    xlabel('Time (sec)');
    ylabel('Eye position (deg)');
    legend('show');
    legend('Hor', 'Ver');
    hold off;
end
end

function [newOnsets, newOffsets] = RemoveFakeSaccades(time, ...
    onsets, offsets, stitchCriteria, minDuration, maxDuration, velocity)

    samplingRate = 1/diff(time(1:2));
    stitchCriteriaSamples = round(stitchCriteria * samplingRate / 1000);
    minDurationSamples = round(minDuration * samplingRate / 1000);
    maxDurationSamples = round(maxDuration * samplingRate / 1000);
    
    % if two consecutive saccades are closer than deltaStitch, merge them
    tempOnsets = onsets;
    tempOffsets = offsets;

    % loop until there is no pair of consecutive saccades closer than
    % stitchCriteria 
    while true
        for c=1:min(length(onsets),length(offsets))-1
            if (onsets(c+1)-offsets(c))<(stitchCriteriaSamples)
                tempOnsets(c+1) = -1;
                tempOffsets(c) = -1;
            end
        end

        s_on = tempOnsets(tempOnsets ~= -1);
        s_off = tempOffsets(tempOffsets ~= -1);
        if sum((s_on(2:end)-s_off(1:end-1)) < (stitchCriteriaSamples))==0
            break;
        end
    end

    newOnsets = tempOnsets(tempOnsets ~= -1);
    newOffsets = tempOffsets(tempOffsets ~= -1);

    if ~isempty(newOnsets) && ~isempty(newOffsets)
        if newOnsets(1)==1
            newOnsets(1) = 2;
        end

        if newOffsets(end) == length(time)
            newOffsets(end) = length(time)-1;
        end
    
    
        % remove too brief and too long saccades
        tooBrief = (newOffsets - newOnsets) < minDurationSamples;
        tooLong = (newOffsets - newOnsets) > maxDurationSamples;
        nanNeighbors = isnan(sum(velocity(newOffsets+1,:),2)) | isnan(sum(velocity(newOnsets-1,:),2));
        discardThis = tooBrief | tooLong | nanNeighbors;
        
        newOnsets(discardThis) = [];
        newOffsets(discardThis) = [];
    end

end

function saccades = GetSaccadeProperties(eyePosition,time,onsets,offsets,...
    velocity,secondaryThreshold)

    hor = eyePosition(:,1);
    ver = eyePosition(:,2);

    % preallocate memory
    saccades = repmat(GetEmptySaccadeStruct, length(onsets),1);

    for i=1:length(onsets)
        [newOnset, newOffset, peakVelocity, meanVelocity] = ...
            ReviseOnsetOffset(time,onsets(i),offsets(i),velocity,secondaryThreshold);

        if ~(isempty(newOnset) || isempty(newOffset) || isempty(peakVelocity))
            
            % extract saccade parameters
            saccades(i).onsetTime = time(newOnset);
            saccades(i).offsetTime = time(newOffset);
            saccades(i).onsetIndex = newOnset;
            saccades(i).offsetIndex = newOffset;
            saccades(i).duration = time(newOffset) - time(newOnset);
            saccades(i).xStart = hor(newOnset);
            saccades(i).xEnd = hor(newOffset);
            saccades(i).yStart = ver(newOnset);
            saccades(i).yEnd = ver(newOffset);
            saccades(i).xAmplitude = hor(newOffset) - hor(newOnset);
            saccades(i).yAmplitude = ver(newOffset) - ver(newOnset);
            saccades(i).vectorAmplitude = sqrt((hor(newOffset) - hor(newOnset)).^2 +...
                (ver(newOffset) - ver(newOnset)).^2);
            saccades(i).direction = atan2d((ver(newOffset) - ver(newOnset)), ...
                (hor(newOffset) - hor(newOnset)));
            saccades(i).peakVelocity = peakVelocity;
            saccades(i).meanVelocity = meanVelocity;
            saccades(i).maximumExcursion = ...
                max(sqrt((hor(newOnset:newOffset) - repmat(hor(newOnset),newOffset-newOnset+1,1)).^2 +...
                (ver(newOnset:newOffset) - repmat(ver(newOnset),newOffset-newOnset+1,1)).^2));
        end
    end
end

function saccade = GetEmptySaccadeStruct
    saccade.duration = [];
    saccade.onsetIndex = [];
    saccade.offsetIndex = [];
    saccade.onsetTime = [];
    saccade.offsetTime = [];
    saccade.xStart = [];
    saccade.xEnd = [];
    saccade.yStart = [];
    saccade.yEnd = [];
    saccade.xAmplitude = [];
    saccade.yAmplitude = [];
    saccade.vectorAmplitude = [];
    saccade.direction = [];
    saccade.peakVelocity = [];
    saccade.meanVelocity = [];
    saccade.maximumExcursion = [];
end

function drift = GetEmptyDriftStruct
    drift.duration = [];
    drift.onsetIndex = [];
    drift.offsetIndex = [];
    drift.onsetTime = [];
    drift.offsetTime = [];
    drift.xStart = [];
    drift.xEnd = [];
    drift.yStart = [];
    drift.yEnd = [];
    drift.xAmplitude = [];
    drift.yAmplitude = [];
    drift.vectorAmplitude = [];
    drift.direction = [];
    drift.peakVelocity = [];
    drift.meanVelocity = [];
    drift.maximumExcursion = [];
end

function [newOnset, newOffset, peakVel, meanVel] = ...
    ReviseOnsetOffset(time,onset,offset,velocity,secondaryThreshold)

    d = 10;

    if onset-d<1
        onset = 1;
    else
        onset = onset - d;
    end
    
    if offset+d>length(time)
        offset = length(time);
    else
        offset = offset+d;
    end

    vectorialVelocity = sqrt(sum(velocity(onset:offset,:).^2,2));
    [peakVel,peakDelta] = max(vectorialVelocity);
    meanVel = nanmean(vectorialVelocity);

    onsetDelta = find(vectorialVelocity(1:peakDelta)<secondaryThreshold,1,'last');
    offsetDelta = find(vectorialVelocity(peakDelta:end)<secondaryThreshold,1,'first');

    newOnset = onset + onsetDelta;
    newOffset = onset + peakDelta + offsetDelta;

    % handle error cases
    if isempty(newOffset)
        newOffset = offset;
    end

    if isempty(newOnset)
        newOnset = onset;
    end

    if newOnset < 0
        newOnset = 1;
    end
    if newOffset > length(time)
        newOffset = length(time);
    end
end

function [onsets, offsets] = GetEventOnsetsAndOffsets(eventIndices)

    % take the difference of indices computed above to find the onset and
    % offset of the movement
    dabove = [0; diff(eventIndices)];
    onsets = find(dabove == 1);
    offsets = find(dabove == -1);

    % make sure we have an offset for every onset.
    if length(onsets) > length(offsets)
        offsets = [offsets; length(eventIndices)];
    elseif length(onsets) < length(offsets)
        offsets = offsets(1:end-1);
    end
end

function [onsets, offsets] = GetDriftOnsetsAndOffsets(eventIndices)

    % take the difference of indices computed above to find the onset and
    % offset of the movement
    dabove = [1; diff(eventIndices)];
    onsets = find(dabove == 1);
    offsets = find(dabove == -1);

    % make sure we have an offset for every onset.
    if length(onsets) > length(offsets)
        offsets = [offsets; length(eventIndices)];
    elseif length(onsets) < length(offsets)
        offsets = offsets(1:end-1);
    end
end

function drifts = GetDriftProperties(eyePosition,time,onsets,offsets,velocity)

    hor = eyePosition(:,1);
    ver = eyePosition(:,2);
    vectorialVelocity = sqrt(sum(velocity.^2,2));

    % preallocate memory
    drifts = repmat(GetEmptyDriftStruct, length(onsets),1);

    for i=1:length(onsets)
        currentOnset = onsets(i);
        currentOffset = offsets(i);
        
        peakVelocity = max(vectorialVelocity(currentOnset:currentOffset));
        meanVelocity = max(vectorialVelocity(currentOnset:currentOffset));
            
        % extract saccade parameters
        drifts(i).onsetTime = time(currentOnset);
        drifts(i).offsetTime = time(currentOffset);
        drifts(i).onsetIndex = currentOnset;
        drifts(i).offsetIndex = currentOffset;
        drifts(i).duration = time(currentOffset) - time(currentOnset);
        drifts(i).xStart = hor(currentOnset);
        drifts(i).xEnd = hor(currentOffset);
        drifts(i).yStart = ver(currentOnset);
        drifts(i).yEnd = ver(currentOffset);
        drifts(i).xAmplitude = hor(currentOffset) - hor(currentOnset);
        drifts(i).yAmplitude = ver(currentOffset) - ver(currentOnset);
        drifts(i).vectorAmplitude = sqrt((hor(currentOffset) - hor(currentOnset)).^2 +...
            (ver(currentOffset) - ver(currentOnset)).^2);
        drifts(i).direction = atan2d((ver(currentOffset) - ver(currentOnset)), ...
            (hor(currentOffset) - hor(currentOnset)));
        drifts(i).peakVelocity = peakVelocity;
        drifts(i).meanVelocity = meanVelocity;
        drifts(i).maximumExcursion = ...
            max(sqrt((hor(currentOnset:currentOffset) - repmat(hor(currentOnset),currentOffset-currentOnset+1,1)).^2 +...
            (ver(currentOnset:currentOffset) - repmat(ver(currentOnset),currentOffset-currentOnset+1,1)).^2));
    end
end


function [driftOnsets, driftOffsets] = MergeDrifts(driftOnsets,driftOffsets,stitchCriteria,time)

    samplingRate = 1/diff(time(1:2));
    stitchCriteriaSamples = round(stitchCriteria * samplingRate / 1000);

    gaps = driftOnsets(2:end) - driftOffsets(1:end-1);
    
    
    for i=1:length(gaps)
        if gaps(i) < stitchCriteriaSamples
            driftOnsets(i+1) = NaN;
            driftOffsets(i) = NaN;
        end
    end
    
    driftOnsets(isnan(driftOnsets)) = [];
    driftOffsets(isnan(driftOffsets)) = [];
end
