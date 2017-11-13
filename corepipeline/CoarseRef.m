function coarseRefFrame = CoarseRef(inputVideoPath, parametersStructure)
%CoarseRef    Generates a coarse reference frame.
%   f = CoarseRef(filename, parametersStructure) is the coarse reference 
%   frame of a video, generated using a scaled down version of each frame 
%   (each frame is scaled down by params.scalingFactor) and then 
%   cross-correlating each of those scaled frames with an arbitrary frame. 
%   If no frame number is provided, the function chooses the middle 
%   frame as the default initial reference frame. The function then 
%   multiplies the scaled down frame shifts by the reciprocal of 
%   scalingFactor to get the actual frame shifts. It then constructs the 
%   coarse reference frame using those approximate frame shifts.
%
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  scalingFactor       :   the amount by which each frame is scaled down
%                          for the coarse cross-correlating
%                          0 < scalingFactor <= 1
%  refFrameNumber      :   specifies which frame is to be used as the 
%                          initial reference frame for the coarse 
%                          cross-correlating. By default, refFrameNumber 
%                          is set to a frame in the middle of the video                        
%  enableVerbosity     :   set to 1 to see the coarseRefFrame, correlation
%                          maps, raw and final eye position traces. Set to 
%                          0 for no feedback.
%  overwrite           :   set to 1 to overwrite existing files resulting 
%                          from calling coarseRefFrame.
%                          Set to 0 to abort the function call if the
%                          files exist in the current directory.
%  enableGPU           :   optional--set to 1 to use the GPU, if 
%                          available, to compute correlation maps. 
%  peakDropWindow      :   size of the correlation map to exclude when
%                          finding the second peak, for calculating peak 
%                          ratios. Units are in pixels. Default value is 20
%                          (20x20 pixel window will be dropped).
%  rotateCorrection    :   optional--set to 1 to call RotateCorrect 
%                          function, which rotates frames and uses rotated 
%                          frames to generate the coarseRefFrame if those 
%                          rotated frames return better peakRatios than the 
%                          un-rotated franes.
%  rotateMaximumPeakRatio : specifies the peak ratio threshold below which 
%                           a rotated frame will be considered a "good" 
%                           frame. This parameter is only necessary if 
%                           rotateCorrection is enabled.
%   
%   Note: CoarseRef also calls StripAnalysis to generate coarse eye 
%   eye position traces. See StripAnalysis for relevant parameters.
%
%   Example usage: 
%       videoPath = 'MyVid.avi';
%       parametersStructure.enableGPU = false;
%       parametersStructure.overwrite = true;
%       parametersStructure.refFrameNumber = 15;
%       parametersStructure.enableVerbosity = 1;
%       parametersStructure.scalingFactor = 0.5;
%       coarseReferenceFrame = CoarseRef(filename, parametersStructure);

%% Initialize variables
outputFileName = [inputVideoPath(1:end-4) '_coarseref'];
outputTracesName = [inputVideoPath(1:end-4) '_coarseframepositions'];
[videoInputArray, videoFrameRate] = VideoPathToArray(inputVideoPath);
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

%% Identify which frames are bad frames
nameEnd = inputVideoPath(1:end-4);
blinkFramesPath = [nameEnd '_blinkframes.mat'];
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end
%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'peakDropWindow')
    parametersStructure.peakDropWindow = 20;
end

if ~isfield(parametersStructure, 'rotateMaximumPeakRatio')
    parametersStructure.rotateMaximumPeakRatio = 0.6;
end

if ~isfield(parametersStructure, 'scalingFactor')
    scalingFactor = 0.5;
else
    scalingFactor = parametersStructure.scalingFactor;
    if ~IsPositiveRealNumber(scalingFactor)
        error('scalingFactor must be a positive, real number');
    end
end

% If no frame number is designated as the original reference frame, then
% the default frame should be the "middle" frame of the total frames (i.e.,
% if frameRate = 30 Hz and duration is 2 seconds, there are 60 total frames
% and the default frame should therefore be the 30th frame).
if ~isfield(parametersStructure, 'refFrameNumber')
    refFrameNumber = floor(size(videoInputArray,3)/2);
else
   refFrameNumber = parametersStructure.refFrameNumber; 
end

if exist('badFrames', 'var')
    while any(badFrames == refFrameNumber)
        if refFrameNumber > 1
            refFrameNumber = refFrameNumber - 1;
        else
            % If this case is ever reached, that means that half the video
            % has been marked as bad frames.
            error(['Parameters for finding blink frames are too strict.', ...
               ' Try setting less stringent parameters, or check the video',...
               ' quality. It may not be of sufficient quality for this',...
               ' program to analyze meaningfully.']);
        end
    end
end

enableGPU = isfield(parametersStructure, 'enableGPU') && ...
    islogical(parametersStructure.enableGPU) && ...
    parametersStructure.enableGPU && ...
    gpuDeviceCount > 0;

%% Shrink video, call strip analysis, and calculate estimated traces.

% Shrink each frame.
shrunkFrames = imresize(videoInputArray, scalingFactor);

% Prepare parameters for calling StripAnalysis, using each shrunk frame as
% a single "strip"
temporaryRefFrame = shrunkFrames(:,:,refFrameNumber);
params = parametersStructure;
params.stripHeight = size(shrunkFrames, 1);
params.stripWidth = size(shrunkFrames, 2);
params.samplingRate = videoFrameRate;
params.badFrames = badFrames;

% Check if user has rotateCorrection enabled.
if isfield(params, 'rotateCorrection') && params.rotateCorrection == true
    [coarseRefFrame, ~] = RotateCorrect(shrunkFrames, videoInputArray, ...
        temporaryRefFrame, outputFileName, params);
    return
else
    [~, usefulEyePositionTraces, ~, ~] = StripAnalysis(shrunkFrames, ...
        temporaryRefFrame, params);
end
framePositions = usefulEyePositionTraces;

%% Remove NaNs in framePositions
try
    % Remove NaNs at beginning and end. Linear interpolation for NaNs in 
    % between, done manually in a custom helper function.
    [filteredStripIndices1, endNaNs1, beginNaNs1] = FilterStrips(framePositions(:, 1));
    [filteredStripIndices2, endNaNs2, beginNaNs2] = FilterStrips(framePositions(:, 2));
    endNaNs = max(endNaNs1, endNaNs2);
    beginNaNs = max(beginNaNs1, beginNaNs2);
    
    framePositions = [filteredStripIndices1 filteredStripIndices2];
    save(outputTracesName, 'framePositions');
catch
    RevasError(outputFileName, 'There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n', parametersStructure);
    error('There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n');
end

%% Scale the coordinates back up.
framePositions = framePositions * 1/scalingFactor;

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

ending = size(usefulEyePositionTraces, 1);

for i = 1+beginNaNs:ending-endNaNs
    frameNumber = i;
    if any(badFrames==frameNumber)
        continue
    else
        framePositionIndex = i - beginNaNs;
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
        
        minRow = round(framePositions(framePositionIndex, 2));
        minColumn = round(framePositions(framePositionIndex, 1));
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
        % Do not show again to GUI since Strip Analysis already showed an
        % uncropped version with yellow eye position traces.
    else
        figure('Name', 'Coarse Reference Frame');
        imshow(coarseRefFrame);
    end
end
end
