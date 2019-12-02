function [saccades, drifts, labels] = FindSaccadesAndDrifts(inputEyePositions,  params)
%FIND SACCADES AND DRIFTS Records in a mat file an array of structures
%representing saccades and an array of structures representing drifts.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputEyePositions| is the path to the eye positions or the eye traces
%   matrix itself with time array. In the former situation, the result is stored with
%   '_sacsdrifts' appended to the input video file name. No result is saved
%   in the latter situation; the result is returned.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite               : set to true to overwrite existing files.
%                             Set to false to abort the function call if the
%                             files already exist. (default false)
%   enableVerbosity         : set to true to report back plots during execution.
%                             (default false)
%   pixelSize               : pixel size in arcminutes. 
%   stitchCriteria          : if two consecutive saccades are closer in time
%                             less than this value (in ms), then they are
%                             stitched back to back. (default 15)
%   minAmplitude            : minimum (micro)saccade amplitude in deg (default 0.1)
%   maxDuration             : maximum saccade duration in ms. (default 100)
%   minDuration             : minimum saccade duration in ms. (default 8)
%   velocityMethod          : set to 1 for regular differentiation. set to 2
%                             for three-point differentiation. (default 2)
%   detectionMethod         : saccade detection method. set to 1 for using a
%                             hard velocity threshold. set to 2 for using a
%                             median-based velocity threshold. (default 2)
%   hardVelocityThreshold   : in deg/sec. (relevant only when detectionMethod is
%                             1) (default 25)
%   hardSecondaryVelocityThreshold : in deg/sec. (relevant only when
%                            detectionMethod is 1) (default 15)
%   thresholdValue          : multiplier specifying the median-based velocity
%                             threshold. (relevant only when detectionMethod
%                             is 2) (default 6)
%   secondaryThresholdValue : multiplier specifying a secondary velocity
%                             threshold for more accurate detection of onset
%                             and offset of a saccade. (relevant only when
%                             detectionMethod is 2) (default 3)
%   axesHandles             : axes handle for giving feedback. if not
%                             provided or empty, new figures are created.
%                             (relevant only when enableVerbosity is true)%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputPath = 'MyFile.mat';
%       params.overwrite = true;
%       params.enableVerbosity = true;
%       params.pixelSize = 10*60/512;
%       params.thresholdValue = 6;
%       params.secondaryThresholdValue = 3;
%       params.stitchCriteria = 15;
%       params.minAmplitude = 0.05;
%       params.minDuration = 8;
%       params.maxDuration = 100;
%       params.detectionMethod = 2;
%       FindSaccadesAndDrifts(inputPath,  params);

%% Determine inputVideo type.
if ischar(inputEyePositions)
    % A path was passed in.
    % Read and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
end

%% Handle overwrite scenarios.
if writeResult
    outputFilePath = Filename(inputEyePositions, 'sacsdrifts');
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~isfield(params, 'overwrite') || ~params.overwrite
        RevasWarning(['FindSaccadesAndDrifts() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        return;
    else
        RevasWarning(['FindSaccadesAndDrifts() is proceeding and overwriting an existing file. (' outputFilePath ')'], params);
    end
end

%% Set parameters to defaults if not specified.
if ~isfield(params, 'pixelSize')
    pixelSize = 10*60/512;
    RevasWarning('using default parameter for pixel size', params);
else
    pixelSize = params.pixelSize;
end

% This second method is the EK Algorithm (Engbert & Kliegl, 2003 Vision Research).
if ~isfield(params, 'thresholdValue')
    lambda = 6;
    RevasWarning('using default parameter for lambda', params);
else
    lambda = params.thresholdValue;
end

if ~isfield(params, 'secondaryThresholdValue')
    secondaryLambda = 3;
    RevasWarning('using default parameter for secondaryLambda', params);
else
    secondaryLambda = params.secondaryThresholdValue;
end

% if enabled, instead of a single threshold, we use a running (adaptive)
% velocity threshold that changes with the variabiltiy of velocity
if ~isfield(params, 'isAdaptive')
    isAdaptive = false;
    RevasWarning('using default parameter for isAdaptive', params);
else
    isAdaptive = params.isAdaptive;
end

% the default form of Engbert & Kliegl algorithm uses median based standard
% deviation. but here we have the ability to use regular sd.
if ~isfield(params, 'isMedianBased')
    isMedianBased = true;
    RevasWarning('using default parameter for isMedianBased', params);
else
    isMedianBased = params.isMedianBased;
end


% units are in milliseconds
if ~isfield(params, 'stitchCriteria')
    stitchCriteria = 15;
    RevasWarning('using default parameter for stitchCriteria', params);
else
    stitchCriteria = params.stitchCriteria;
end

% units are in degrees
if ~isfield(params, 'minAmplitude')
    minAmplitude = 0.1;
    RevasWarning('using default parameter for minAmplitude', params);
else
    minAmplitude = params.minAmplitude;
end

% units are in milliseconds
if ~isfield(params, 'minDuration')
    minDuration = 8;
    RevasWarning('using default parameter for minDuration', params);
else
    minDuration = params.minDuration;
end

% units are in milliseconds
if ~isfield(params, 'maxDuration')
    maxDuration = 100;
    RevasWarning('using default parameter for maxDuration', params);
else
    maxDuration = params.maxDuration;
end

% Method to use for detection
% 1 = Hard velocity threshold
% 2 = Median-based (default)
if ~isfield(params, 'detectionMethod')
    detectionMethod = 2;
else
    detectionMethod = params.detectionMethod;
end

% units are in degrees/second
if ~isfield(params, 'hardVelocityThreshold')
    hardVelocityThreshold = 25;
    if detectionMethod == 1
        RevasWarning('using default parameter for hardVelocityThreshold', params);
    end
else
    hardVelocityThreshold = params.hardVelocityThreshold;
end

% units are in degrees/second
if ~isfield(params, 'hardSecondaryVelocityThreshold')
    hardSecondaryVelocityThreshold = 15;
    if detectionMethod == 1
        RevasWarning('using default parameter for hardSecondaryVelocityThreshold', params);
    end
else
    hardSecondaryVelocityThreshold = params.hardVelocityThreshold;
end

% Method to use to calculate velocity
% 1 = using |diff|
% 2 = (x_(n+1) - x_(n-1)) / 2 delta t)
if ~isfield(params, 'velocityMethod')
    velocityMethod = 2;
else
    velocityMethod = params.velocityMethod;
end

% check verbosity field
if ~isfield(params, 'enableVerbosity')
    enableVerbosity = false;
else
    enableVerbosity = params.enableVerbosity;
end

% save parameters structure in the environment before loading a different
% parameters structure
axesHandlesFlag = isfield(params, 'axesHandles');
if axesHandlesFlag
    axesHandles = params.axesHandles;
end

%% Load mat file with output from |StripAnalysis|
if writeResult
    load(inputEyePositions,'eyePositionTraces', 'timeArray');
else
    eyePositionTraces = inputEyePositions(1:end, 1:end-1);
    timeArray = inputEyePositions(1:end, end:end);
end

%% Convert eye position traces from pixels to degrees
eyePositionTraces = eyePositionTraces * (pixelSize/60);  %#ok<*NODEF>

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

if detectionMethod == 1
    velocityThreshold = hardVelocityThreshold;
    secondaryVelocityThreshold = hardSecondaryVelocityThreshold;
else
    medianVelocity = nanmedian(vectorialVelocity);
    if isAdaptive
        sdVelocity = AdaptiveThreshold(vectorialVelocity,timeArray,isMedianBased);
    else
        if isMedianBased
            sdVelocity = sqrt(nanmedian(vectorialVelocity.^2) - medianVelocity.^2);
        else
            sdVelocity = nanstd(vectorialVelocity);
        end
    end
    velocityThreshold = medianVelocity + lambda * sdVelocity;
    secondaryVelocityThreshold = medianVelocity + secondaryLambda * sdVelocity;
end

% commented out by MNA on 11/22/18. 
% % it's enough to exceed the threshold in one dimension only
% % saccadeIndices = velocity > velocityThreshold;
% % saccadeIndices = saccadeIndices(:,1) | saccadeIndices(:,2);

% use vectorial velocity for more robust detection % MNA 11/22/18
saccadeIndices = vectorialVelocity > velocityThreshold;

% compute saccade onset and offset indices
[onsets, offsets] = GetEventOnsetsAndOffsets(saccadeIndices);

% remove artifacts
[onsets, offsets] = RemoveFakeSaccades(timeArray, ...
    onsets, offsets, stitchCriteria, minDuration, maxDuration, vectorialVelocity);

% Get saccade properties
saccades = GetSaccadeProperties(eyePositionTraces,timeArray,onsets,offsets,...
    vectorialVelocity, secondaryVelocityThreshold);

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

% compute drift onsets and offsets
[driftOnsets, driftOffsets] = GetDriftOnsetsAndOffsets(driftIndices);

% if the temporal gap between two consecutive drifts is less than minimum
% saccade duration, then stitch.
[driftOnsets, driftOffsets] = MergeDrifts(driftOnsets,driftOffsets,minDuration,timeArray);

% get drift parameters
drifts = GetDriftProperties(eyePositionTraces,timeArray,driftOnsets,driftOffsets,velocity); 

labels = ~driftIndices; % 1: saccade, 0: drift

%% Save to output mat file.
if writeResult
    save(outputFilePath, 'saccades', 'drifts', 'labels','params');
end

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
    ax.ColorOrderIndex = 1;
    plot(timeArray(~driftIndices), eyePositionTraces(~driftIndices,1),'o','LineWidth',2); hold on;
    plot(timeArray(~driftIndices), eyePositionTraces(~driftIndices,2),'o','LineWidth',2); hold on;
    title('Eye position');
    xlabel('Time (sec)');
    ylabel('Eye position (deg)');
    legend('show');
    legend('Hor', 'Ver');
    hold off;
end
end



% get a running (adaptive) velocity threshold depending on the variability
% in the velocity
function sds = AdaptiveThreshold(velocity,timeArray,isMedianBased)

if nargin < 3
    isMedianBased = 0;
end

samplingRate = round(1/diff(timeArray(1:2)));
windowSize = round(1*samplingRate); % 1 second
stepSize = round(0.1*samplingRate); % 100ms

sds = nan(size(timeArray));
steps = 1 : stepSize : length(timeArray)-windowSize;
for i = 1:length(steps)-1
    ix = steps(i):(steps(i)+windowSize-1);
    if isMedianBased
        sds(ix) = sqrt(nanmedian(velocity(ix).^2) - nanmedian(velocity(ix))^2);
    else
        sds(ix) = nanstd(velocity(ix));
    end
end

% in case we have nans at the end, replace them with the nearest nonnan
% value
lastNonNan = find(~isnan(sds),1,'last');
sds(lastNonNan:end) = sds(lastNonNan);


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

function saccades = GetSaccadeProperties(eyePosition,timeArray,onsets,offsets,...
    velocity,secondaryThreshold)

    hor = eyePosition(:,1);
    ver = eyePosition(:,2);

    % preallocate memory
    saccades = repmat(GetEmptySaccadeStruct, length(onsets),1);

    for i=1:length(onsets)
        [newOnset, newOffset, peakVelocity, meanVelocity] = ...
            ReviseOnsetOffset(timeArray,onsets(i),offsets(i),velocity,secondaryThreshold);

        if ~(isempty(newOnset) || isempty(newOffset) || isempty(peakVelocity))
            
            % extract saccade parameters
            saccades(i).onsetTime = timeArray(newOnset);
            saccades(i).offsetTime = timeArray(newOffset);
            saccades(i).onsetIndex = newOnset;
            saccades(i).offsetIndex = newOffset;
            saccades(i).duration = timeArray(newOffset) - timeArray(newOnset);
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
    ReviseOnsetOffset(timeArray,onset,offset,vectorialVelocity,secondaryThreshold)

    d = 10;

    if onset-d<1
        onset = 1;
    else
        onset = onset - d;
    end
    
    if offset+d>length(timeArray)
        offset = length(timeArray);
    else
        offset = offset+d;
    end

    %vectorialVelocity = sqrt(sum(velocity(onset:offset,:).^2,2));
    vectorialVelocity = vectorialVelocity(onset:offset);
    secondaryThreshold = secondaryThreshold(onset:offset);
    [peakVel,peakDelta] = max(vectorialVelocity);
    meanVel = nanmean(vectorialVelocity);

    onsetDelta = find(vectorialVelocity(1:peakDelta)<secondaryThreshold(1:peakDelta),1,'last');
    offsetDelta = find(vectorialVelocity(peakDelta:end)<secondaryThreshold(peakDelta:end),1,'first');

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
    if newOffset > length(timeArray)
        newOffset = length(timeArray);
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
