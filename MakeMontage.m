function [refFrame] = MakeMontage(params, fileName)
% MakeMontage    Reference frame.
%   MakeMontage(params, fileName) generates a reference frame by averaging
%   all the pixel values across all frames in a video.
%   
%   Note: The params variable should have the attributes params.stripHeight,
%   params.positions, params.time and params.samplingRate. 
%   params.newStripHeight is an optional parameter.
%
%   Example: 
%       params.stripHeight = 15;
%       params.time = 1/540:1/540:1;
%       params.positions = randn(540, 2);
%       params.samplingRate = 540;
%       MakeMontage(params,'fileName');

if isfield(params, 'newStripHeight')
    % if params.newStripHeight is a field but no value is specified, then set
    % params.newStripHeight equal to params.stripHeight
    if isempty(params.newStripHeight)
        params.newStripHeight = params.stripHeight;
    end
    
    % if params.newStripHeight is not a field, set params.newStripHeight
    % equal to params.stripHeight anyway
else
    params.newStripHeight = params.stripHeight;
end
    
stripIndices = params.positions;
t1 = params.time;

% grabbing info about the video and strips
videoInfo = VideoReader(fileName);
frameHeight = videoInfo.Height;
w = videoInfo.Width;
frameRate = videoInfo.Framerate;
totalFrames = frameRate * videoInfo.Duration;
stripsPerFrame = round(frameHeight/params.newStripHeight);

% setting up templates for reference frame and counter array
counterArray = zeros(frameHeight*2);
refFrame = zeros(frameHeight*2);

% Negate all positions 
stripIndices = -stripIndices;

% Scale the frame coordinates so that all values are positive. Take the
% negative value with the highest magnitude in each column of
% framePositions and add that value to all the values in the column. This
% will zero the most negative value (like taring a weight scale). Then add
% a positive integer to make sure all values are > 0 (i.e., if we leave the
% zero'd value in framePositions, there will be indexing problems later,
% since MatLab indexing starts from 1 not 0).
mostNegative = max(-1*stripIndices);
stripIndices(:, 1) = stripIndices(:, 1) + mostNegative(1) + 2;
stripIndices(:, 2) = stripIndices(:, 2) + mostNegative(2) + 2;

% scaling the time array to accomodate new strip height CHANGE TOTALFRAMES
% TO FRAMERATE MAYBE
scalingFactor = ((params.stripHeight)/2)/(totalFrames*frameHeight);
t1 = t1 + scalingFactor;
dt = params.newStripHeight / (totalFrames * frameHeight);
t2 = dt:dt:videoInfo.Duration + scalingFactor;

% Make sure that both time arrays have the same dimensions
if size(t2, 1) ~= size(t1, 1) && size(t2, 2) ~= size(t1, 2)
    t2 = t2';
end

% replace NaNs with a linear interpolation, done manually in a helper
% function
filteredStripIndices1 = FilterStrips(stripIndices(:, 1));
filteredStripIndices2 = FilterStrips(stripIndices(:, 2));
filteredStripIndices = [filteredStripIndices1 filteredStripIndices2];

% interpolating positions between strips, using pchip 
interpolatedPositions = interp1(t1, filteredStripIndices, t2, 'pchip');

for frameNumber = 1:totalFrames
    
    % Read frame and convert pixel values to signed integers
    videoFrame = double(readFrame(videoInfo))/255;
    
    % get the appropriate strips from stripIndices for each frame
    n = frameNumber;
    startFrameStrips = round(1 + ((n-1)*(stripsPerFrame)));
    endFrameStrips = round(n * stripsPerFrame);
    
    if endFrameStrips > size(interpolatedPositions, 1)
        endFrameStrips = size(interpolatedPositions, 1);
    end
    
    % keep track of the strip number, so we can move it vertically accordingly
    stripNumber = 1;

    for strip = startFrameStrips : endFrameStrips
        % row and column "coordinates" of the top left pixel of each strip
        topLeft = [interpolatedPositions(strip, 1), interpolatedPositions(strip, 2)];
        columnIndex = round(topLeft(2));
        rowIndex = round(topLeft(1));
        
        % move strip to proper position
        rowIndex = rowIndex + ((stripNumber-1) * params.newStripHeight);
        
        % get max row/column of the strip
        maxRow = rowIndex + params.newStripHeight - 1;
        maxColumn = columnIndex + w - 1;
        
        % transfer values of the strip pixels to the reference frame, and
        % increment the corresponding location on the counter array
        templateSelectRow = rowIndex:maxRow;
        templateSelectColumn = columnIndex:maxColumn;
        vidStart = ((stripNumber-1)*params.newStripHeight)+1;
        vidEnd = stripNumber * params.newStripHeight;
        
        % If the strip extends beyond the frame (i.e., the frame has a
        % height of 512 pixels and strip of height 10 begins at row 511)
        % set the max row of that strip to be the last row of the frame.
        % Also make templateSelectRow smaller to match the dimensions of
        % the new strip
         if vidEnd > size(videoFrame, 1)
             difference = vidEnd - size(videoFrame, 1);
             vidEnd = size(videoFrame, 1);
             templateSelectRow = rowIndex:(maxRow-difference);
         end
        
        refFrame(templateSelectRow, templateSelectColumn) = refFrame(...
            templateSelectRow, templateSelectColumn) + videoFrame(...
            vidStart:vidEnd, :);
        counterArray(templateSelectRow, templateSelectColumn) = counterArray...
            (templateSelectRow, templateSelectColumn) + 1;

        % increment stripNumber for the next iteration of the for-loop
        stripNumber = stripNumber + 1;
    end
end


% divide each pixel in refFrame by the number of strips that contain that pixel
refFrame = refFrame./counterArray;

% Crop out the leftover 0 padding from the original template.
column1 = interpolatedPositions(:, 1);
column2 = interpolatedPositions(:, 2);
minRow = min(column1);
minColumn = min(column2);
maxColumn = max(column2);
refFrame(1:floor((minRow-1)), :) = [];
refFrame(:, 1:floor((minColumn-1))) = [];
refFrame(:, ceil(maxColumn+w):end) = [];

% Convert any NaN values in the reference frame to a 0. Otherwise, running
% strip analysis on this new frame will not work
NaNindices = find(isnan(refFrame));
for k = 1:size(NaNindices)
    NaNindex = NaNindices(k);
    refFrame(NaNindex) = 0;
end

% Need to take care of rows separately for cropping out 0 padding because 
% strip locations do not give info about where the template frame ends
k = 1;
while k<= size(refFrame, 1)
    if refFrame(k, :) == 0
        refFrame(k, :) = [];
        continue
    end
    k = k + 1;
end

% UNCOMMENT THE SAVE STATEMENT IN THE FINAL VERSION
% save('Reference Frame', 'refFrame');
figure(1)
imshow(refFrame);

end