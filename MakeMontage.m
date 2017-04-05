function [refFrame] = MakeMontage(params, fileName)
% MakeMontage takes in a structure params, which contains information about
% each strip (i.e., strip height, location, etc), and the name of the file. 
% Using the information in params, MakeMontage constructs a reference frame 
% using the average of all the strips.
% params must have params.stripHeight, params.usefulEyePositionTraces, and
% params.SamplingRate. 


%isfield 
stripIndices = params.usefulEyePositionTraces;

% grabbing info about the video and strips
videoInfo = VideoReader(fileName);
frameHeight = videoInfo.Height;
w = videoInfo.Width;
frameRate = videoInfo.Framerate;
totalFrames = frameRate * videoInfo.Duration;
stripsPerFrame = round(params.SamplingRate/frameRate);
stripInterval = round(frameHeight / stripsPerFrame);

% setting up templates for reference frame and counter array
counterArray = zeros(frameHeight*2);
refFrame = zeros(frameHeight*2);

% creating another video object in order to use each individual frame later
videoFReader = vision.VideoFileReader(fileName);

% scale the strip coordinates so that all values are positive
mostNegative = max(-1*stripIndices);
stripIndices(:, 1) = stripIndices(:, 1) + mostNegative(1) + 1;
stripIndices(:, 2) = stripIndices(:, 2) + mostNegative(2) + 1;

for frameNumber = 1:totalFrames
    
    videoFrame = step(videoFReader);
    
    % get the appropriate strips from stripIndices for each frame
    n = frameNumber;
    startFrameStrips = 1 + ((n-1)*(stripsPerFrame));
    endFrameStrips = n * stripsPerFrame;
    
    % keep track of the strip number, so we can move it vertically
    % accordingly
    stripNumber = 1;
    
    for strip = startFrameStrips : endFrameStrips
    
        % if one of the coordinates is NaN, skip that iteration
        if isnan(stripIndices(strip, 1)) || isnan(stripIndices(strip, 2))
            
            % increment stripNumber for the next iteration of the for-loop
            stripNumber = stripNumber + 1;
            continue
        end
        
        % row and column "coordinates" of the top left pixel of each strip
        topLeft = [stripIndices(strip, 1), stripIndices(strip, 2)];
        columnIndex = round(topLeft(2));
        rowIndex = round(topLeft(1));
        
        % move strip to proper position assuming all strips are evenly
        % spaced
        rowIndex = rowIndex + ((stripNumber-1) * stripInterval);
       
        % transfer values of the strip pixels to the reference frame
        for i = rowIndex : rowIndex + params.stripHeight -1
            for j = columnIndex : columnIndex + w - 1
                refFrame(i,j) = refFrame(i,j) + videoFrame(i,j);
            end
        end
    
        % increment the corresponding location on the counter array
        counterArray(rowIndex:rowIndex+stripInterval-1, columnIndex:columnIndex+(w-1))...
        = counterArray(rowIndex:rowIndex+stripInterval-1, columnIndex:columnIndex+(w-1))... 
        + 1;
    
        % increment stripNumber for the next iteration of the for-loop
        stripNumber = stripNumber + 1;
    end
end

% divide each pixel in refFrame by the number of strips that contain that pixel
refFrame = refFrame./counterArray;

imshow(refFrame);
end