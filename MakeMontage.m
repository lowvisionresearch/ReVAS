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

% creating another video object in order to use each individual frame later
videoFReader = vision.VideoFileReader(fileName);

% scale the strip coordinates so that all values are positive
mostNegative = max(-1*stripIndices);
stripIndices(:, 1) = stripIndices(:, 1) + mostNegative(1) + 2;
stripIndices(:, 2) = stripIndices(:, 2) + mostNegative(2) + 2;

% scaling the time array to accomodate new strip height
scalingFactor = ((params.stripHeight)/2)/(frameRate*frameHeight);
t1 = t1 + scalingFactor;
dt = params.newStripHeight / (frameRate * frameHeight);
t2 = 0:dt:videoInfo.Duration + scalingFactor;

% transpose t2 because t1 is a column vector
t2 = t2';

% replace NaNs with a linear interpolation, done manually in a helper
% function
filteredStripIndices1 = FilterStrips(stripIndices(:, 1));
filteredStripIndices2 = FilterStrips(stripIndices(:, 2));
filteredStripIndices = [filteredStripIndices1 filteredStripIndices2];

% interpolating positions between strips, using pchip
interpolatedPositions = interp1(t1, filteredStripIndices, t2, 'pchip');

for frameNumber = 1:totalFrames
    
    videoFrame = step(videoFReader);
    
    % get the appropriate strips from stripIndices for each frame
    n = frameNumber;
    startFrameStrips = round(1 + ((n-1)*(stripsPerFrame)));
    endFrameStrips = round(n * stripsPerFrame);
    
    % keep track of the strip number, so we can move it vertically
    % accordingly
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
        
        % making sure the row values don't exceed videoFrame dimensions
        if maxRow > frameHeight
            maxRow = frameHeight;
        end
      
        % transfer values of the strip pixels to the reference frame, and
        % increment the corresponding location on the counter array
        for i = rowIndex : maxRow
            for j = columnIndex : maxColumn
                refFrame(i,j) = refFrame(i,j) + videoFrame(i,j);
                counterArray(i, j) = counterArray(i, j) + 1;
            end
        end

        % increment stripNumber for the next iteration of the for-loop
        stripNumber = stripNumber + 1;
    end
end

% divide each pixel in refFrame by the number of strips that contain that pixel
refFrame = refFrame./counterArray;

refFrame = Crop(refFrame, w*2, frameHeight * 2, 100);
imshow(refFrame);

end