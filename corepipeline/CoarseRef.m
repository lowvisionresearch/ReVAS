function coarseRefFrame = CoarseRef(inputVideo, parametersStructure)
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
%   |inputVideo| is the path to the video or a matrix containing the video,
%   or a matrix containing the video.
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
%       inputVideo = 'MyVid.avi';
%       load('MyVid_params.mat')
%       coarseReferenceFrame = CoarseRef(inputVideo, coarseParameters);

%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
    writeResult = true;
    inputVideoPath = inputVideo;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
    inputVideoPath = '';
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Initialize variables
if writeResult
    outputFilePath = Filename(inputVideo, 'coarseref');
    outputTracesPath = [inputVideo(1:end-4) '_coarseframepositions.mat'];
    blinkFramesPath = Filename(inputVideo, 'blink');
    shrunkFilePath = [inputVideo(1:end-4) '_shrunk.avi'];
else
    outputTracesPath = fullfile(pwd, '.coarseframepositions.mat');
    blinkFramesPath = fullfile(pwd, '.blinkframes.mat');
end

%% Handle overwrite scenarios.
if writeResult
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasWarning(['CoarseRef() did not execute because it would overwrite existing file. (' outputFilePath ')'], parametersStructure);
        try
            RevasWarning(['Loading ''coarseRefFrame'' from (' outputFilePath ')'], parametersStructure);
            load(outputFilePath,'coarseRefFrame');
        catch
            RevasError(inputVideoPath,'Loading ''coarseRefFrame'' failed. Returning an empty array!!!', parametersStructure);
            coarseRefFrame = [];
        end
        return;
    else
        RevasWarning(['CoarseRef() is proceeding and overwriting an existing file. (' outputFilePath ')'], parametersStructure);  
    end
end

%% Identify which frames are bad frames
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'peakDropWindow')
    parametersStructure.peakDropWindow = 20;
    RevasWarning('using default parameter for peakDropWindow', parametersStructure);
end

if ~isfield(parametersStructure, 'rotateMaximumPeakRatio')
    parametersStructure.rotateMaximumPeakRatio = 0.6;
    RevasWarning('using default parameter for rotateMaximumPeakRatio', parametersStructure);
end

if ~isfield(parametersStructure, 'scalingFactor')
    parametersStructure.scalingFactor = 1;
    RevasWarning('using default parameter for scalingFactor', parametersStructure);
else
    if ~IsPositiveRealNumber(parametersStructure.scalingFactor)
        error('scalingFactor must be a positive, real number');
    end
end

% If no frame number is designated as the original reference frame, then
% the default frame should be the 3rd frame. (arbitrary decision at this
% point)
if ~isfield(parametersStructure, 'refFrameNumber')
    parametersStructure.refFrameNumber = 3;
    RevasWarning('using default parameter for refFrameNumber', parametersStructure);
end

if ~isfield(parametersStructure, 'frameIncrement')
    parametersStructure.frameIncrement = 1;
    RevasWarning('using default parameter for frameIncrement', parametersStructure);
end

if exist('badFrames', 'var')
    while any(badFrames == parametersStructure.refFrameNumber)
        if parametersStructure.refFrameNumber > 1
            parametersStructure.refFrameNumber = parametersStructure.refFrameNumber - 1;
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

if ~writeResult && ~isfield(parametersStructure, 'FrameRate')
    parametersStructure.FrameRate = 30;
    RevasWarning('using default parameter for FrameRate', parametersStructure);
end

parametersStructure.enableGPU = isfield(parametersStructure, 'enableGPU') && ...
    islogical(parametersStructure.enableGPU) && ...
    parametersStructure.enableGPU && ...
    gpuDeviceCount > 0;

%% Shrink video, call strip analysis, and calculate estimated traces.

if writeResult
    reader = VideoReader(inputVideo);
    parametersStructure.FrameRate = reader.FrameRate;
    numberOfFrames = parametersStructure.FrameRate * reader.Duration;
    height = reader.Height;

else
    % parametersStructure.FrameRate set in "Set parameters to default if not specified"
    % section already.
    [height, ~, numberOfFrames] = size(inputVideo);
end

% Use a shrunk video iff scalingFactor is specified to not be 1.
if parametersStructure.scalingFactor ~= 1
    try
        % Try and check whether the shrunk video already exists

        % Since we're not writing files, we should not try to load a shrunk
        % video. Just skip to the catch block to make a new one.
        if ~writeResult
            error
        end

        % If overwrite is enabled, do not use a previously made shrunk video.
        % Make a new one in the catch block below.
        if parametersStructure.overwrite
            error
        end

        shrunkReader = VideoReader(shrunkFilePath);
        % If it does exist, check that it's of correct dimensions. If the
        % shrunk video is not scaled to the desired amount, throw an error to
        % create a new shrunkVideo in the subsequent catch block.
        if shrunkReader.Height/reader.Height ~= parametersStructure.scalingFactor
            error
        else
            frameNumber = 1;
            while hasFrame(shrunkReader)
                frame = readFrame(shrunkReader);
                if ndims(frame) == 3
                    frame = rgb2gray(frame);
                end
                if frameNumber == parametersStructure.refFrameNumber
                    temporaryRefFrame = frame;
                    break;
                end
                frameNumber = frameNumber + 1;
            end
        end

    catch
        if writeResult
            writer = VideoWriter(shrunkFilePath, 'Grayscale AVI');
            open(writer);
        else
            % preallocate shrunk video
            shrunkVideo = zeros( ...
                size(inputVideo, 1) * parametersStructure.scalingFactor, ...
                size(inputVideo, 2) * parametersStructure.scalingFactor, ...
                size(inputVideo, 3), ...
                'uint8');
        end

        for frameNumber = 1:numberOfFrames

            if writeResult
                frame = readFrame(reader);
                if ndims(frame) == 3
                    frame = rgb2gray(frame);
                end
            else
                frame = inputVideo(1:end, 1:end, frameNumber);
            end

            if rem(frameNumber,parametersStructure.frameIncrement) == 0
                frame = imresize(frame, parametersStructure.scalingFactor);

                % Sometimes resizing causes numbers to dip below 0 (but just barely)
                frame(frame<0) = 0;
                % Similarly, values occasionally exceed 255
                frame(frame>255) = 255;

                if writeResult
                    writeVideo(writer, frame);
                else
                    shrunkVideo(1:end, 1:end, frameNumber) = frame;
                end
            end

            if frameNumber == parametersStructure.refFrameNumber
                temporaryRefFrame = frame;
            end        
        end

        if writeResult
            close(writer);
        end
    end
    
else
    % Even if we do not need a shrunk video, we still need to grab the
    % user's desired temporary reference frame.
    
    if writeResult
        for frameNumber = 1:numberOfFrames
            % Keep reading frames until we hit the user's desired temporary
            % reference frame number, then break.
            
            temporaryRefFrame = readFrame(reader);
            
            if frameNumber == parametersStructure.refFrameNumber
                if ndims(temporaryRefFrame) == 3
                    temporaryRefFrame = rgb2gray(temporaryRefFrame);
                end
                break;
            end
        end
    else
        temporaryRefFrame = inputVideo(1:end, 1:end, parametersStructure.refFrameNumber);
    end
end

% Prepare parameters for calling StripAnalysis, using each shrunk frame as
% a single "strip"
params = parametersStructure;

if parametersStructure.scalingFactor == 1
    % The shrunk video was not made above in this case, since it is the
    % same as the inputVideo.
    shrunkVideo = inputVideo;
elseif writeResult
    shrunkReader = VideoReader(shrunkFilePath);
    params.stripHeight = shrunkReader.Height;
    params.stripWidth = shrunkReader.Width;   
    shrunkVideo = shrunkFilePath;
else
    [params.stripHeight, params.stripWidth, ~] = size(shrunkVideo);
end

params.samplingRate = parametersStructure.FrameRate;
params.badFrames = badFrames;
params.maximumPeakRatio = 0.8;
params.minimumPeakThreshold = 0.2;

try
    if ndims(temporaryRefFrame) == 3
        temporaryRefFrame = rgb2gray(temporaryRefFrame);
    end
catch
    RevasError(inputVideoPath, ...
        'Your chosen reference frame number is out of bounds.', ...
        parametersStructure);
    coarseRefFrame = [];
    return;
end
% Check if user has rotateCorrection enabled.
if isfield(params, 'rotateCorrection') && params.rotateCorrection
    [coarseRefFrame, ~] = RotateCorrect(shrunkVideo, inputVideo, ...
        temporaryRefFrame, outputFilePath, params);
    return;
else
    [~, usefulEyePositionTraces, ~, ~] = StripAnalysis(shrunkVideo, ...
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
    if isempty(beginNaNs)
        beginNaNs = 0;
    end
    if isempty(endNaNs)
        beginNaNs = 0;
    end
    
    framePositions = [filteredStripIndices1 filteredStripIndices2];
    save(outputTracesPath, 'framePositions');
catch
    RevasError(inputVideoPath, 'There were no useful eye position traces. Lower the minimumPeakThreshold and/or raise the maximumPeakRatio.\n', parametersStructure);
end

if ~exist('beginNaNs','var')
      beginNaNs = 0;
end
if ~exist('endNaNs','var')
      endNaNs = 0;
end

%% Scale the coordinates back up.
framePositions = framePositions * 1/parametersStructure.scalingFactor;

%% Set up the counter array and the template for the coarse reference frame.
counterArray = zeros(height*3);
coarseRefFrame = zeros(height*3);

framePositions = round(ScaleCoordinates(framePositions));

if parametersStructure.enableGPU
    numberOfFrames = gpuArray(numberOfFrames);
    framePositions = gpuArray(framePositions);
    counterArray = gpuArray(counterArray);
    coarseRefFrame = gpuArray(coarseRefFrame);
end

ending = size(usefulEyePositionTraces, 1);

if writeResult
   reader = VideoReader(inputVideo);
end

for frameNumber = 1:numberOfFrames
    
    if writeResult
        frame = readFrame(reader);
        if ndims(frame) == 3
            frame = rgb2gray(frame);
        end
    else
        frame = inputVideo(1:end, 1:end, frameNumber);
    end
    
    if frameNumber < (1 + beginNaNs) || any(badFrames == frameNumber)
        continue;
    elseif frameNumber > ending-endNaNs 
        break;
    end
    framePositionIndex = frameNumber - beginNaNs;
    
    if parametersStructure.enableGPU
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
end

% Divide the template frame by the counterArray to obtain the average value
% for each pixel.
coarseRefFrame = coarseRefFrame./counterArray;

if parametersStructure.enableGPU
    coarseRefFrame = gather(coarseRefFrame);
end

coarseRefFrame = uint8(coarseRefFrame);

%% Remove extra padding from the coarse reference frame
coarseRefFrame = Crop(coarseRefFrame);

if writeResult
    save(outputFilePath, 'coarseRefFrame');
end

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
