function coarseRefFrame = CoarseRef(params, scalingFactor)
%CoarseRef    Generates a coarse reference frame.
%   f = CoarseRef(params, scalingFactor) is the coarse reference frame of a
%   video, generated using a scaled down version of each frame (each frame
%   is scaled down by scalingFactor) and then cross-correlating each of 
%   those scaled frames with an arbitrary frame number. If no frame number
%   is provided, the function chooses the middle frame as the default
%   initial reference frame. The function then multiplies the scaled down
%   frame shifts by the reciprocal of scalingFactor to get the actual frame
%   shifts. It then constructs the coarse reference frame using those
%   approximate frame shifts.
%
%   params must have the field params.fileName, which is the video that
%   will be analyzed. params.refFrameNumber is an optional parameter that
%   designates which frame to use as the initial scaled down reference
%   frame. scalingFactor is the factor by which each frame will be
%   multiplied to get the approximate frame shifts.
%
%   Example: 
%       params.fileName = 'MyVid.avi';
%       params.refFrameNumber = 15;
%       scalingFactor = 0.5;
%       CoarseRef(params, scalingFactor);


% get video info
v = VideoReader(params.fileName);
totalFrames = v.frameRate * v.Duration;

% Preallocate an array to store frames because it is expensive to change
% the size of an array in a loop--assumes that all video frames have the
% same dimensions. In order to preallocate, the size of each frame must
% first be known. This is why the following lines examine the size of the
% first frame of the video after shrinking. timeToRemember lets us "rewind"
% the video later after reading one frame here.
timeToRemember = v.CurrentTime;
sampleFrame = readFrame(v);
shrunkFrame = imresize(sampleFrame, scalingFactor);
[rows, columns] = size(shrunkFrame);
frames = zeros(rows, columns*totalFrames);

frameNumber = 1;
v.CurrentTime = timeToRemember;
while hasFrame(v)
    currFrame = readFrame(v);
    shrunkFrame = imresize(currFrame, scalingFactor);
    
    % startIndex/endIndex indicate where to insert each frame into the
    % preallocated array. For example, the second frame (frameNumber = 2)
    % will be inserted at frames(:, 257:512) and the third frame
    % (frameNumber = 3) will be inserted at frames(:, 513:768) if the frame 
    % size is 256x256. Each frame is added to the array "horizontally."
    startIndex = ((frameNumber-1) * columns) + 1;
    endIndex = frameNumber*columns;
    frames(:, startIndex:endIndex) = shrunkFrame;
    frameNumber = frameNumber + 1;
end

% if no frame number is designated as the original reference frame, then
% the default frame should be the "middle" frame of the total frames (i.e.,
% if frameRate = 30 Hz and duration is 2 seconds, there are 60 total frames
% and the default frame should therefore be the 30th frame).
if ~isfield(params, 'refFrameNumber')
    params.refFrameNumber = totalFrames/2;
end

% Same logic for choosing values for startIndex/endIndex as before. This
% time we need to extract the correct frame from the frames array.
startIndex = ((params.refFrameNumber-1) * columns) + 1;
endIndex = params.refFrameNumber*columns;
temporaryRefFrame = frames(:, startIndex:endIndex);

% Preallocating an array for frame positions. framePositions will contain
% the frame shifts relative to temporaryRefFrame
framePositions = zeros(totalFrames, 2);

% Perform cross-correlation using the temporary reference frame
frameNumber = 1;
while frameNumber <= totalFrames
    startIndex = ((frameNumber-1) * columns) + 1;
    endIndex = frameNumber*columns;
    currFrame = frames(:, startIndex:endIndex);
    c = normxcorr2(temporaryRefFrame, currFrame);

    % Find peak in cross-correlation.
    [ypeak, xpeak] = find(c==max(c(:)));
    
    % Only perform the next steps if there is a reasonable peak. For
    % example, if the correlation map is a flat plane, skip this iteration.
    if size([ypeak, xpeak]) == [1, 2]
       
        % Account for the padding that normxcorr2 adds.
        yoffSet = ypeak-size(currFrame,1);
        xoffSet = xpeak-size(currFrame,2);
        
        % Use the peak to get the top-left coordinate of the frame relative 
        % to the temporary reference frame. 
        columnCoordinate = xoffSet + 1;
        rowCoordinate = yoffSet + 1;
        
        % Scale back up. Currently the program is using the shrunken frames
        % as a rough reference. Therefore, the positions need to be
        % multiplied by the reciprocal of the scaling factor to get the
        % real frame movements (i.e., if a frame is shrunken by 1/2 and
        % moves 6 units to the right of the shrunken reference frame, then 
        % the real frame moved 12 units to the right relative to the real 
        % reference frame).
        framePositions(frameNumber, 1) = (1/scalingFactor) * rowCoordinate;
        framePositions(frameNumber, 2) = (1/scalingFactor) * columnCoordinate;
    else
        framePositions(frameNumber, :) = [];
    end
   
    frameNumber = frameNumber + 1;
    
end

% Set up the counter array and the template for the coarse reference frame.
height = size(sampleFrame, 1);
counterArray = zeros(height*2);
coarseRefFrame = zeros(height*2);

% Scale the frame coordinates so that all values are positive. Take the
% negative value with the highest magnitude in each column of
% framePositions and add that value to all the values in the column. This
% will zero the most negative value (like taring a weight scale). Then add
% a positive integer to make sure all values are > 0 (i.e., if we leave the
% zero'd value in framePositions, there will be indexing problems later,
% since MatLab indexing starts from 1 not 0).
mostNegative = max(-1*framePositions);
framePositions(:, 1) = framePositions(:, 1) + mostNegative(1) + 2;
framePositions(:, 2) = framePositions(:, 2) + mostNegative(2) + 2;

% Create a new video object using VideoFileReader because the original 
% video object has integers of a class that can't be added to the 
% template matrix. 
newVidObj = vision.VideoFileReader(params.fileName);
frameNumber = 1;

for f = 1:totalFrames
    frame = step(newVidObj);
    
    % framePositions has the top left coordinate of the frames, so those
    % coordinates will represent the minRow and minColumn to be added to
    % the template frame. maxRow and maxColumn will be the size of the
    % frame added to the minRow/minColumn - 1. (i.e., if size of the frame
    % is 256x256 and the minRow is 1, then the maxRow will be 1 + 256 - 1.
    % If minRow is 2 (moved to the right by one pixel) then maxRow will be 
    % 2 + 256 - 1 = 257)
    minRow = framePositions(frameNumber, 1);
    minColumn = framePositions(frameNumber, 2);
    maxRow = size(frame, 1) + minRow - 1;
    maxColumn = size(frame, 2) + minColumn - 1;
    
    % Now add the frame values to the template frame and increment the
    % counterArray, which is keeping track of how many frames are added 
    % to each pixel. Need to loop through each color since newVidObj is RGB
    selectRow = round(minRow):round(maxRow);
    selectColumn = round(minColumn):round(maxColumn);
    for color = 1:3
    coarseRefFrame(selectRow, selectColumn) = coarseRefFrame(selectRow, ...
        selectColumn) + frame(:, :, color);
    counterArray(selectRow, selectColumn) = counterArray(selectRow, selectColumn) + 1;
    end
    
    frameNumber = frameNumber + 1;
end

% Divide the template frame by the counterArray to obtain the average value
% for each pixel.
coarseRefFrame = coarseRefFrame./counterArray;

% Crop out the leftover 0 padding from the original template.
coarseRefFrame = Crop(coarseRefFrame, height*2, height * 2, 100);

imshow(coarseRefFrame)

end