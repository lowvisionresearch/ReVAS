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
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  enableVerbosity     :   set to true to see the coarseRefFrame, correlation
%                          maps, raw and final eye position traces (default
%                          false)
%  overwrite           :   set to true to overwrite existing files.
%                          Set to false to abort the function call if the
%                          files already exist. (default false)
%  scalingFactor       :   the amount by which each frame is scaled down
%                          for the coarse cross-correlating
%                          0 < scalingFactor <= 1 (default 1)
%  refFrameNumber      :   specifies which frame is to be used as the 
%                          initial reference frame for the coarse 
%                          cross-correlating. (default totalFrames / 2)                        
%  enableGPU           :   optional--set to true to use the GPU, if 
%                          available, to compute correlation maps (default
%                          false)
%  peakDropWindow      :   size of the correlation map to exclude when
%                          finding the second peak, for calculating peak 
%                          ratios. Units are in pixels. (default 20)
%  rotateCorrection    :   optional--set to true to call RotateCorrect 
%                          function, which rotates frames and uses rotated 
%                          frames to generate the coarseRefFrame if those 
%                          rotated frames return better peakRatios than the 
%                          un-rotated franes (default false)
%  rotateMaximumPeakRatio : specifies the peak ratio threshold below which 
%                           a rotated frame will be considered a "good" 
%                           frame. This parameter is only necessary if 
%                           rotateCorrection is enabled (default 0.6)
%   
%   Note: CoarseRef also calls StripAnalysis to generate coarse eye 
%   eye position traces. See StripAnalysis for relevant parameters.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideoPath = 'MyVid.avi';
%       load('MyVid_params.mat')
%       coarseReferenceFrame = CoarseRef(inputVideoPath, coarseParameters);

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Initialize variables
outputFileName = [inputVideoPath(1:end-4) '_coarseref'];
outputTracesName = [inputVideoPath(1:end-4) '_coarseframepositions'];
reader = VideoReader(inputVideoPath);
videoFrameRate = reader.Framerate;
numberOfFrames = videoFrameRate * reader.Duration;

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
    scalingFactor = 1;
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
    refFrameNumber = floor(numberOfFrames / 2);
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
            RevasError(inputVideoPath, ['Parameters for finding blink frames are too strict.', ...
                ' Try setting less stringent parameters, or check the video',...
                ' quality. It may not be of sufficient quality for this',...
                ' program to analyze meaningfully.'], parametersStructure);
        end
    end
end

enableGPU = isfield(parametersStructure, 'enableGPU') && ...
    islogical(parametersStructure.enableGPU) && ...
    parametersStructure.enableGPU && ...
    gpuDeviceCount > 0;

%% Shrink video, call strip analysis, and calculate estimated traces.

shrunkFileName = [inputVideoPath(1:end-4) '_shrunk.avi'];

% First check whether the shrunk video already exists
try
    shrunkReader = VideoReader(shrunkFileName);
    % If it does exist, check that it's of correct dimensions. If the
    % shrunk video is not scaled to the desired amount, throw an error to
    % create a new shrunkVideo in the subsequent catch block.
    if shrunkReader.Height/reader.Height ~= parametersStructure.scalingFactor
        error
    else
        frameNumber = 1;
        while hasFrame(shrunkReader)
            frame = readFrame(shrunkReader);
            if frameNumber == refFrameNumber
                temporaryRefFrame = frame;
                break;
            end
            frameNumber = frameNumber + 1;
        end
    end
catch
    writer = VideoWriter(shrunkFileName, 'Grayscale AVI');
    open(writer);
    
    frameNumber = 1;
    while hasFrame(reader)
        frame = readFrame(reader);
        frame = imresize(frame, scalingFactor);
        
        % Sometimes resizing causes numbers to dip below 0 (but just barely)
        frame(frame<0) = 0;
        % Similarly, values occasionally exceed 255
        frame(frame>255) = 255;
        
        writeVideo(writer, frame);
        if frameNumber == refFrameNumber
            temporaryRefFrame = frame;
        end
        frameNumber = frameNumber + 1;
    end
    close(writer);
end

% Prepare parameters for calling StripAnalysis, using each shrunk frame as
% a single "strip"
shrunkReader = VideoReader(shrunkFileName);
params = parametersStructure;
params.stripHeight = shrunkReader.Height;
params.stripWidth = shrunkReader.Width;
params.samplingRate = videoFrameRate;
params.badFrames = badFrames;
params.originalVideoPath = shrunkFileName;

% Check if user has rotateCorrection enabled.
if isfield(params, 'rotateCorrection') && params.rotateCorrection
    [coarseRefFrame, ~] = RotateCorrect(shrunkFileName, inputVideoPath, ...
        temporaryRefFrame, outputFileName, params);
    return;
else
    [~, usefulEyePositionTraces, ~, ~] = StripAnalysis(shrunkFileName, ...
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
    RevasError(inputVideoPath, 'There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n', parametersStructure);
end

%% Scale the coordinates back up.
framePositions = framePositions * 1/scalingFactor;

%% Set up the counter array and the template for the coarse reference frame.
height = reader.Height;
counterArray = zeros(height*3);
coarseRefFrame = zeros(height*3);

framePositions = round(ScaleCoordinates(framePositions));

if enableGPU
    numberOfFrames = gpuArray(numberOfFrames);
    framePositions = gpuArray(framePositions);
    counterArray = gpuArray(counterArray);
    coarseRefFrame = gpuArray(coarseRefFrame);
end

ending = size(usefulEyePositionTraces, 1);
frameNumber = 1;
reader = VideoReader(inputVideoPath);

while hasFrame(reader)
    frame = readFrame(reader);
    if frameNumber < (1 + beginNaNs) || any(badFrames == frameNumber)
        frameNumber = frameNumber + 1;
        continue;
    elseif frameNumber > ending-endNaNs 
        break;
    end
    framePositionIndex = frameNumber - beginNaNs;
    
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
        selectColumn) + double(frame);
    counterArray(selectRow, selectColumn) = counterArray(selectRow, selectColumn) + 1;
    frameNumber = frameNumber + 1;
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
