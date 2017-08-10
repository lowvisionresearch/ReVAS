function coarseRefFrame = CoarseRef(videoPath, parametersStructure)
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
%   parametersStructure must have the fields: parametersStructure.scalingFactor,
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
%       parametersStructure.scalingFactor = 0.5;
%       CoarseRef(parametersStructure, filename);

outputFileName = [videoPath(1:end-4) '_coarseref'];
outputTracesPath = [videoPath(1:end-4) '_coarseframepositions'];

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
nameEnd = strfind(videoPath,'dwt_');
blinkFramesPath = [videoPath(1:nameEnd+length('dwt_')-1) 'blinkframes'];
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end

%% Check to see if operations can be performed on GPU and whether the
% user wants to do so if there is a GPU
enableGPU = parametersStructure.enableGPU & (gpuDeviceCount > 0);

%% Shrink video, call strip analysis, and restore video's original scale.

% Shrink each frame and write to a new video so that stripAnalysis can be
% called, using each frame as one "strip"
[videoInputArray, videoFrameRate] = VideoPathToArray(videoPath);
shrunkFrames = imresize(videoInputArray, parametersStructure.scalingFactor);

% if no frame number is designated as the original reference frame, then
% the default frame should be the "middle" frame of the total frames (i.e.,
% if frameRate = 30 Hz and duration is 2 seconds, there are 60 total frames
% and the default frame should therefore be the 30th frame).
if ~isfield(parametersStructure, 'refFrameNumber')
    refFrameNumber = floor(size(shrunkFrames,3)/2);
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
[~, usefulEyePositionTraces, ~, ~] = StripAnalysis(shrunkFrames, ...
    temporaryRefFrame, params);

% Scale the coordinates back up.
framePositions = ...
    usefulEyePositionTraces * 1/parametersStructure.scalingFactor;

%% Remove NaNs in framePositions

% Remove NaNs at beginning and end.
% Interpolate for NaNs in between.
[filteredStripIndices1, ~, ~] = FilterStrips(framePositions(:, 1));
[filteredStripIndices2, ~, ~] = FilterStrips(framePositions(:, 2));
framePositions = [filteredStripIndices1 filteredStripIndices2];
save(outputTracesPath, 'framePositions');

%% Set up the counter array and the template for the coarse reference frame.
v = VideoReader(videoPath);
frameRate = v.FrameRate;
totalFrames = size(framePositions, 1);

height = v.Height;
counterArray = zeros(height*3);
coarseRefFrame = zeros(height*3);

% Negate all frame positions
framePositions = -framePositions;

% Scale the strip coordinates so that all values are positive. Take the
% negative value with the highest magnitude in each column of
% framePositions and add that value to all the values in the column. This
% will zero the most negative value (like taring a weight scale). Then add
% a positive integer to make sure all values are > 0 (i.e., if we leave the
% zero'd value in framePositions, there will be indexing problems later,
% since MatLab indexing starts from 1 not 0).
column1 = framePositions(:, 1);
column2 = framePositions(:, 2); 

if column1(column1<0)
   mostNegative = max(-1*column1);
   framePositions(:, 1) = framePositions(:, 1) + mostNegative + 2;
end

if column2(column2<0)
    mostNegative = max(-1*column2);
    framePositions(:, 2) = framePositions(:, 2) + mostNegative + 2;
end

if column1(column1<0.5)
    framePositions(:, 1) = framePositions(:, 1) + 2;
end

if column2(column2<0.5)
    framePositions(:, 2) = framePositions(:, 2) + 2;
end

if any(column1==0)
    framePositions(:, 1) = framePositions(:, 1) + 2;
end

if any(column2==0)
    framePositions(:, 2) = framePositions(:, 2) + 2;
end

if enableGPU
    totalFrames = gpuArray(totalFrames);
    framePositions = gpuArray(framePositions);
    counterArray = gpuArray(counterArray);
    coarseRefFrame = gpuArray(coarseRefFrame);
end

for frameNumber = 1:totalFrames
    if any(badFrames==frameNumber)
        continue
    else
        % Use double function because readFrame gives unsigned integers,
        % whereas we need to use signed integers
        if enableGPU
            frame = double(gpuArray(readFrame(v)))/255;
        else
            frame = double(readFrame(v))/255;
        end

        % framePositions has the top left coordinate of the frames, so those
        % coordinates will represent the minRow and minColumn to be added to
        % the template frame. maxRow and maxColumn will be the size of the
        % frame added to the minRow/minColumn - 1. (i.e., if size of the frame
        % is 256x256 and the minRow is 1, then the maxRow will be 1 + 256 - 1.
        % If minRow is 2 (moved down by one pixel) then maxRow will be 
        % 2 + 256 - 1 = 257)

        minRow = round(framePositions(frameNumber, 2));
        minColumn = round(framePositions(frameNumber, 1));
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
% Convert any NaN values in the reference frame to a 0. Otherwise, running
% strip analysis on this new frame will not work
NaNindices = find(isnan(coarseRefFrame));
for k = 1:size(NaNindices)
    NaNindex = NaNindices(k);
    coarseRefFrame(NaNindex) = 0;
end

% Crop out the leftover 0 padding from the original template. First check
% for 0 rows
k = 1;
while k<=size(coarseRefFrame, 1)
    if coarseRefFrame(k, :) == 0
        coarseRefFrame(k, :) = [];
        continue
    end
    k = k + 1;
end

% Then sweep for 0 columns
k = 1;
while k<=size(coarseRefFrame, 2)
    if coarseRefFrame(:, k) == 0
        coarseRefFrame(:, k) = [];
        continue
    end
    k = k + 1;
end

save(outputFileName, 'coarseRefFrame');

if parametersStructure.enableVerbosity >= 1
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(3));
    else
        figure('Name', 'Coarse Reference Frame');
    end
    imshow(coarseRefFrame)
end
end