function coarseRefFrame = CoarseRef(inputVideoPath, parametersStructure)
%CoarseRef    Generates a coarse reference frame.
%   f = CoarseRef(filename, parametersStructure) is the coarse reference 
%   frame of a video, generated using a scaled down version of each frame 
%   (each frame is scaled down by params.scalingFactor) and then 
%   cross-correlating each of those scaled frames with an arbitrary frame 
%   number. If no frame number is provided, the function chooses the middle 
%   frame as the default initial reference frame. The function then 
%   multiplies the scaled down frame shifts by the reciprocal of 
%   scalingFactor to get the actual frame shifts. It then constructs the 
%   coarse reference frame using those approximate frame shifts.
%
%   parametersStructure must have the fields: scalingFactor,
%   parametersStructure.refFrameNumber, 
%   an optional parameter that designates which frame to use as the initial 
%   scaled down reference frame, parametersStructure.overwrite (optional), 
%   parametersStructure.enableVerbosity, which is either 0, 1, or 2. Verbosity of 0 will 
%   only save the output in a MatLab file. Verbosity of 1 will display the 
%   final result. Verbosity of 2 will show the progress of the program. 
%   scalingFactor is the factor by which each frame will be multiplied to 
%   get the approximate frame shifts.
%   parametersStructure also needs parametersStructure.peakRatio,
%   parametersStructure.minimumPeakThreshold.
%
%   Example: 
%       videoPath = 'MyVid.avi';
%       parametersStructure.enableGPU = false;
%       parametersStructure.overwrite = true;
%       parametersStructure.refFrameNumber = 15;
%       parametersStructure.enableVerbosity = 2;
%       scalingFactor = 0.5;
%       CoarseRef(filename, parametersStructure);

outputFileName = [inputVideoPath(1:end-4) '_coarseref'];
outputTracesName = [inputVideoPath(1:end-4) '_coarseframepositions'];

%% Handle overwrite scenarios.
if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['CoarseRef() did not execute because it would overwrite existing file. (' outputFileName ')'], parametersStructure);
    coarseRefFrame = [];
    return;
else
    RevasWarning(['CoarseRef() is proceeding and overwriting an existing file. (' outputFileName ')'], parametersStructure);  
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'scalingFactor')
    scalingFactor = 0.5; % standard deviation of the gaussian kernel, in pixels
    RevasWarning('using default parameter for scalingFactor', parametersStructure);
else
    scalingFactor = parametersStructure.scalingFactor;
    if ~IsPositiveRealNumber(scalingFactor)
        error('scalingFactor must be a positive, real number');
    end
end

% Validation or default assignment on reference frame number is done below
% after the video is imported and the frames are shrunk.

enableGPU = isfield(parametersStructure, 'enableGPU') && ...
    islogical(parametersStructure.enableGPU) && ...
    parametersStructure.enableGPU && ...
    gpuDeviceCount > 0;

%% Identify which frames are bad frames
nameEnd = inputVideoPath(1:end-4);
blinkFramesPath = [nameEnd '_blinkframes.mat'];
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end

%% Shrink video, call strip analysis, and restore video's original scale.

% Shrink each frame and write to a new video so that stripAnalysis can be
% called, using each frame as one "strip"
[videoInputArray, videoFrameRate] = VideoPathToArray(inputVideoPath);
shrunkFrames = imresize(videoInputArray, scalingFactor);

% if no frame number is designated as the original reference frame, then
% the default frame should be the "middle" frame of the total frames (i.e.,
% if frameRate = 30 Hz and duration is 2 seconds, there are 60 total frames
% and the default frame should therefore be the 30th frame).
if ~isfield(parametersStructure, 'refFrameNumber')
    refFrameNumber = floor(size(shrunkFrames,3)/2);
    RevasWarning('using default parameter for refFrameNumber', parametersStructure);
else
   refFrameNumber = parametersStructure.refFrameNumber; 
end
if exist('badFrames', 'var')
    while any(badFrames == refFrameNumber)
        if refFrameNumber > 1
            refFrameNumber = refFrameNumber - 1;
        else
            refFrameNumber = refFrameNumber + 1;
        end
    end
end

temporaryRefFrame = shrunkFrames(:,:,refFrameNumber);
params = parametersStructure;
params.stripHeight = size(shrunkFrames, 1);
params.stripWidth = size(shrunkFrames, 2);
params.samplingRate = videoFrameRate;
params.badFrames = badFrames;

% Check if user has rotateCorrection enabled. Flag will check for torsional 
% movement.
if isfield(params, 'rotateCorrection') && params.rotateCorrection == true
    % RotateCorrect used to be integrated with CoarseRef but is now its own
    % independent function. Too tedious to go back and change the parts of
    % CoarseRef that depended on RotateCorrect, so just stop CoarseRef here
    % if RotateCorrect is enabled.
    params.maximumPeakRatio = params.rotateMaximumPeakRatio;
    [coarseRefFrame, ~] = RotateCorrect(shrunkFrames, videoInputArray, ...
        temporaryRefFrame, outputFileName, params);
    return
else

params.peakDropWindow = 20;
[~, usefulEyePositionTraces, ~, ~] = StripAnalysis(shrunkFrames, ...
    temporaryRefFrame, params);
end

framePositions = usefulEyePositionTraces;
%% Remove NaNs in framePositions
try
    % Remove NaNs at beginning and end.
    % Interpolate for NaNs in between.
    [filteredStripIndices1, endNaNs1, beginningNaNs1] = FilterStrips(framePositions(:, 1));
    [filteredStripIndices2, endNaNs2, beginningNaNs2] = FilterStrips(framePositions(:, 2));
    endNaNs = max(endNaNs1, endNaNs2);
    if endNaNs == -1
        endNaNs = 0;
    end
    beginningNaNs = max(beginningNaNs1, beginningNaNs2);

    framePositions = [filteredStripIndices1 filteredStripIndices2];
    save(outputTracesName, 'framePositions');
catch
    RevasError(outputFileName, 'There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n', parametersStructure);
    error('There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n');
end

%% Scale the coordinates back up.
framePositions = ...
    framePositions * 1/scalingFactor;

%% Set up the counter array and the template for the coarse reference frame.
totalFrames = size(videoInputArray, 3);
height = size(videoInputArray, 1);
counterArray = zeros(height*3);
coarseRefFrame = zeros(height*3);

framePositions = round(ScaleCoordinates(framePositions));

if enableGPU
    totalFrames = gpuArray(totalFrames);
    framePositions = gpuArray(framePositions);
    counterArray = gpuArray(counterArray);
    coarseRefFrame = gpuArray(coarseRefFrame);
end

nonNanIndices = find(~isnan(usefulEyePositionTraces(:,1)));

for i = 1:length(nonNanIndices)
    frameNumber = nonNanIndices(i);
    if any(badFrames==frameNumber)
        continue
    else
        % Use double function because readFrame gives unsigned integers,
        % whereas we need to use signed integers
        frame = double(videoInputArray(:, :, frameNumber))/255;
        
        if enableGPU
            frame = gpuArray(frame);
        end
        % framePositions has the top left coordinate of the frames, so those
        % coordinates will represent the minRow and minColumn to be added to
        % the template frame. maxRow and maxColumn will be the size of the
        % frame added to the minRow/minColumn - 1. (i.e., if size of the frame
        % is 256x256 and the minRow is 1, then the maxRow will be 1 + 256 - 1.
        % If minRow is 2 (moved down by one pixel) then maxRow will be 
        % 2 + 256 - 1 = 257)
        
        minRow = round(framePositions(i, 2));
        minColumn = round(framePositions(i, 1));
        maxRow = size(frame, 1) + minRow - 1;
        maxColumn = size(frame, 2) + minColumn - 1;

        % Now add the frame values to the template frame and increment the
        % counterArray, which is keeping track of how many frames are added 
        % to each pixel. 
        selectRow = round(minRow):round(maxRow);
        selectColumn = round(minColumn):round(maxColumn);

        coarseRefFrame(selectRow, selectColumn) = coarseRefFrame(selectRow, ...
            selectColumn) + frame;
        counterArray(selectRow, selectColumn) = counterArray(selectRow, selectColumn) + 1;
    end
end

% Divide the template frame by the counterArray to obtain the average value
% for each pixel.
coarseRefFrame = coarseRefFrame./counterArray;

if enableGPU
    coarseRefFrame = gather(coarseRefFrame);
end

%% Remove extra padding from the coarse reference frame
coarseRefFrame = Crop(coarseRefFrame);

save(outputFileName, 'coarseRefFrame');

if isfield(parametersStructure, 'enableVerbosity') && ...
        parametersStructure.enableVerbosity >= 1
    if isfield(parametersStructure, 'axesHandles')
        % Not showing cropped version to GUI because Strip Analysis has
        % already shown non-cropped version with yellow traces on top.
    else
        figure('Name', 'Coarse Reference Frame');
        imshow(coarseRefFrame);
    end
end
end