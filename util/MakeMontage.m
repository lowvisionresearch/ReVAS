function [refFrame] = MakeMontage(parametersStructure, fileName)
% MakeMontage    Reference frame.
%   MakeMontage(parametersStructure, fileName) generates a reference frame by averaging
%   all the pixel values across all frames in a video.
%   
%   -----------------------------------
%   Input
%   -----------------------------------
%   |fileName| is the path to the video.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  stripHeight         :   the height of each strip that was used in
%                          StripAnalysis. Units are in pixels (default 15)
%  newStripHeight      :   optional--interpolate between strip positions 
%                          with a finer time interval, using newStripHeight
%                          as the new interval (default stripHeight)
%  time                :   the time array that corresponds to the eye
%                          position traces (no default--this must be specified)
%  positions           :   the eye position traces (no default--this must be 
%                          specified)
%  addNoise            :   optional--add this field to fill black regions
%                          in the final reference frame with random noise. 
%                          FineRef adds this field on the final iteration 
%                          of a given run (default false)
%  stabilizeVideo      :   set to true to generate a stabilized video
%                          (default false)
%  stabilizedVideoSizeMultiplier : a multiplier to determine the frame size
%                          of the stabilized video (default 1.25)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideoPath = 'MyVid.avi';
%       load('MyVid_final.mat')
%       load('MyVid_params.mat')
%       parametersStructure.positions = eyePositionTraces;
%       parametersStructure.time = timeArray;
%       parametersStructure.stripHeight = 15;
%       referenceFrame = MakeMontage(parametersStructure, inputVideoPath);

%% Set default values. 
if ~isfield(parametersStructure, 'stripHeight')
    parametersStructure.stripHeight = 15;
end

% newStripHeight will give the requisite information for interpolation
% between the spaced-out strips later.
if isfield(parametersStructure, 'newStripHeight')
    % if parametersStructure.newStripHeight is a field but no value is specified, then set
    % parametersStructure.newStripHeight equal to parametersStructure.stripHeight
    if isempty(parametersStructure.newStripHeight)
        parametersStructure.newStripHeight = parametersStructure.stripHeight;
    end
    
    % if parametersStructure.newStripHeight is not a field, set parametersStructure.newStripHeight
    % equal to parametersStructure.stripHeight anyway
else
    parametersStructure.newStripHeight = parametersStructure.stripHeight;
end

% Size of the stabilized video will be stabilizedVideoSizeMultiplier*frameSize
if ~isfield(parametersStructure,'stabilizedVideoSizeMultiplier') 
    stabilizedVideoSizeMultiplier = 1.25;
else
    stabilizedVideoSizeMultiplier = parametersStructure.stabilizedVideoSizeMultiplier;
end

%% Identify which frames are bad frames
nameEnd = fileName(1:size(fileName, 2)-4);
blinkFramesPath = [nameEnd '_blinkframes.mat'];
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end

%% Initialize variables
stripIndices = parametersStructure.positions;
t1 = parametersStructure.time;

% Grabbing info about the video and strips
videoInfo = VideoReader(fileName);
frameHeight = videoInfo.Height;
width = videoInfo.Width;
frameRate = videoInfo.Framerate;
totalFrames = frameRate * videoInfo.Duration;
if isfield(parametersStructure, 'stabilizeVideo') && parametersStructure.stabilizeVideo
    stripsPerFrame = ceil(frameHeight/parametersStructure.newStripHeight);
else
    stripsPerFrame = floor(frameHeight/parametersStructure.newStripHeight);
end

% Set up templates for reference frame and counter array
counterArray = zeros(frameHeight*3);
refFrame = zeros(frameHeight*3);

% Prepare a video writer object if user enables option to generate a
% stabilized video
if isfield(parametersStructure, 'stabilizeVideo') && ...
        parametersStructure.stabilizeVideo
    
    stabilizedVideoFilename = fileName;
    stabilizedVideoFilename(end-3:end) = [];
    stabilizedVideoFilename(end+1:end+11) = '_stabilized';
    stabilizedVideo = VideoWriter(stabilizedVideoFilename);
    open(stabilizedVideo)
else
    stabilizedVideo = false;
end

%% Set up the interpolation
% Scale the time array to accomodate new strip height
scalingFactor = ((parametersStructure.stripHeight)/2)/(frameRate*frameHeight);
t1 = t1 + scalingFactor;
dt = parametersStructure.newStripHeight / (frameRate * frameHeight);
t2 = dt:dt:videoInfo.Duration + scalingFactor;

% Make sure that both time arrays have the same dimensions
if size(t2, 1) ~= size(t1, 1) && size(t2, 2) ~= size(t1, 2)
    t2 = t2';
end

%% Remove NaNs in stripIndices

% Replace the rest of the NaNs with linear interpolation, done
% manually in a helper function. NaNs at the end of stripIndices will be
% deleted, along with their corresponding time points.
[filteredStripIndices1, endNaNs1, beginNaNs1] = FilterStrips(stripIndices(:, 1));
[filteredStripIndices2, endNaNs2, beginNaNs2] = FilterStrips(stripIndices(:, 2));
beginNaNs = max(beginNaNs1, beginNaNs2);
endNaNs = max(endNaNs1, endNaNs2);

% Handle the case in which the two column vectors are different sizes.
difference1 = beginNaNs1 - beginNaNs2;
difference2 = endNaNs1 - endNaNs2;

% These next cases may seem counter-intuitive. This is because extra NaNs
% have already been removed, so all that remains is to remove numbers from
% the other column. For example, in this first case, difference1 < 0
% indicates that beginNaNs2 > beginNaNs 1. It might seem like the next
% action would be to remove numbers from filteredStripIndices2 because
% beginNaNs2 > beginNaNs1. However, because beginNaNs2 > beginNaNs1,
% actually all the NaNs in filteredStripIndices2 have already been removed.
% What remains is to remove the non-NaN numbers from filteredStripIndices1.
if difference1 < 0
    difference1 = -difference1;
    filteredStripIndices1(1:difference1, :) = [];
elseif difference1 > 0
    filteredStripIndices2(1:difference1, :) = [];
end

if difference2 < 0
    difference2 = -difference2;
    filteredStripIndices1(end-difference2+1:end, :) = [];
elseif difference2 > 0
    filteredStripIndices2(end-difference2+1:end, :) = [];
end

filteredStripIndices = [filteredStripIndices1 filteredStripIndices2];
if beginNaNs >= 1
    t1(1:beginNaNs) = [];
end

if endNaNs >= 1
    t1(end-endNaNs+1:end) = [];
end

%% Perform interpolation with finer time interval

% interpolating positions between strips
interpolatedPositions = interp1(t1, filteredStripIndices, t2, 'linear');
interpolatedPositions = movmean(interpolatedPositions, 45);

%% Prepare interpolatedPositions for generating the reference frame.

% Add a third column to interpolatedPositions to hold the strip numbers
% because when I filter for NaNs, the strip number method I originally used
% gets thrown off. Add numberOfNaNs because the number you remove from the
% beginning will skew the rest of the data
for stripNumber = beginNaNs + 1:size(interpolatedPositions, 1) + beginNaNs
    interpolatedPositions(stripNumber-beginNaNs, 3) = stripNumber;
end

% Remove leftover NaNs. Need to use while statement because the size of
% interpolatedPositions changes each time I remove an NaN. Adding the
% stripNumbers in the third column earlier will take care of offsets.
k = 1;
while k<=size(interpolatedPositions, 1)
    if isnan(interpolatedPositions(k, 2))
        interpolatedPositions(k, :) = [];
        t2(k) = [];
        continue
    elseif isnan(interpolatedPositions(k, 1))
        interpolatedPositions(k, :) = [];
        t2(k) = [];
        continue
    end
    k = k + 1;
end

% Scale the strip coordinates so that all values are positive. 
interpolatedPositions(:, 1:2) = ScaleCoordinates(interpolatedPositions(:,...
    1:2));

% Round the final scaled positions to get valid matrix indices
interpolatedPositions = round(interpolatedPositions);

% Leftover and skips will be used to correct for rounding errors; see the 
% for-loop below for details 
leftover = (frameHeight/parametersStructure.newStripHeight) - ...
    floor(frameHeight/parametersStructure.newStripHeight);
leftoverCopy = leftover;
skips = 0;

% Only grab center coordinates once (for generating stabilized videos)
foundCenter = false;

%% Use interpolatedPositions to generate the reference frame.
for frameNumber = 1:totalFrames
    
    % By default, the current frame is not one that needs the correction
    % factor for rounding
    correctionFrame = false;
    
    % Read frame.
    videoFrame = readFrame(videoInfo);
    
    % get the appropriate strips from stripIndices for each frame
    startFrameStrips = round(1 + ((frameNumber-1)*(stripsPerFrame))) + skips;
    endFrameStrips = round(frameNumber * stripsPerFrame) + skips;
    frameStripsWithoutNaN = zeros(size(startFrameStrips:endFrameStrips, 2), 3);
    
    % However, keep in mind that there will be errors that accumulate
    % through rounding the start/end frame strip indices. For example, if
    % there are 4 frames of height 10 and a strip height of 3, then there
    % are (supposedly) 3.33333 strips per frame. The first strip of the
    % fourth frame should be the 11th index in interpolatedPositions.
    % Instead, due to rounding, the first strip of the fourth frame will be
    % the 10th index. Here, we handle this rounding error.
    if leftover >= 0.999
        % I chose the threshold to be 0.999 instead of 1 because sometimes
        % MatLab doesn't store enough digits in leftover to get proper
        % numbers (i.e., if frame height is 488 and strip height is 9).
        correctionFrame = true;
        endFrameStrips = endFrameStrips + 1;
        leftover = leftover - 1;
        skips = skips + 1;
    end
    leftover = leftover + leftoverCopy;
    
    % Extract the strip positions that will be used from this frame. Some
    % strips may be missing because NaNs were removed earlier
    for k = 1:size(interpolatedPositions, 1)
        if interpolatedPositions(k, 3) > endFrameStrips
            break
        elseif startFrameStrips <= interpolatedPositions(k, 3) && ...
                interpolatedPositions(k, 3) <= endFrameStrips
            frameStripsWithoutNaN(k, :) = interpolatedPositions(k, :);
        end
    end
    
    % Remove extra 0's from frameStripsWithoutNaN, leftover from the
    % preallocation.
    k = 1;
    while k<=size(frameStripsWithoutNaN, 1)
        if frameStripsWithoutNaN(k, :) == 0
            frameStripsWithoutNaN(k, :) = [];
            continue
        end
        k = k + 1;
    end
    
    % Remove the used interpolatedPositions to speed up run time in later
    % iterations
    numberToRemove = size(frameStripsWithoutNaN, 1);
    if numberToRemove < size(interpolatedPositions, 1)
        interpolatedPositions(1:numberToRemove, :) = [];
    end
    
    % We skip the iteration here instead of at the beginning of the
    % for-loop because the processing for interpolatedPositions and
    % leftOver was necessary for future frames
    if any(badFrames==frameNumber)
        if ~isfield(parametersStructure.stabilizeVideo)...
            || ~parametersStructure.stabilizeVideo
            continue
        else
            writeVideo(stabilizedVideo, refFrame);
        end
    else
        % Add a fourth column to hold the stripNumbers
        if size(frameStripsWithoutNaN, 1) >= 2
            if correctionFrame == true
                frameStripsWithoutNaN(1, 4) = mod(frameStripsWithoutNaN(1, 3)-skips, ...
                    stripsPerFrame) + 1;
            else
                frameStripsWithoutNaN(1, 4) = mod(frameStripsWithoutNaN(1, 3)-skips, ...
                    stripsPerFrame);
            end
            for k = 2:size(frameStripsWithoutNaN, 1)
                % Using the first stripNumber as a reference, add the difference
                % between the two indices. For example, if index 5 has stripNumber
                % of 1 and the next valid index is 8, then the stripNumber for
                % index 8 is (8-5)+1 = 4 (indices 6 and 7, which were
                % presumably NaNs, would have been stripNumbers 2 and 3).
                frameStripsWithoutNaN(k, 4) = (frameStripsWithoutNaN(k, 3)...
                    - frameStripsWithoutNaN(k-1,3)) + frameStripsWithoutNaN(k-1, 4);
            end
        elseif size(frameStripsWithoutNaN, 1) == 1
            % Handle the obscure case in which frameStripsWithoutNaN is only
            % one row, and that one row may be in the frame in which we need to
            % correct for rounding errors.
            if correctionFrame == true
                x = frameStripsWithoutNaN(1, 3);
                y = (stripsPerFrame+1)*skips + (stripsPerFrame*(frameNumber-...
                    skips-1));
                frameStripsWithoutNaN(1, 4) = x - y + 1;
            else
                frameStripsWithoutNaN(1, 4) = mod(frameStripsWithoutNaN(1, 3)-skips, ...
                    stripsPerFrame);
                if frameStripsWithoutNaN(1, 4) == 0
                    frameStripsWithoutNaN(1, 4) = stripsPerFrame;
                end
            end
        end
        
        if size(frameStripsWithoutNaN, 1) >= 1
            % In case the first stripNumber was 0, add one to the rest of the
            % column (should be only in the case when there is only one row)
            if frameStripsWithoutNaN(1, 4) == 0
                frameStripsWithoutNaN(:, 4) = frameStripsWithoutNaN(:, 4) + 1;
            end
        end

        for strip = 1 : size(frameStripsWithoutNaN, 1)
            
            % Reset this boolean--it is used to signal that a frame has
            % been completed for the stabilized video generation
            endOfFrame = false;
            
            % Keep track of the stripNumber so we can shift it accordingly
            stripNumber = frameStripsWithoutNaN(strip, 4);
            
            % row and column "coordinates" of the top left pixel of each strip
            topLeft = [frameStripsWithoutNaN(strip, 1), frameStripsWithoutNaN(strip, 2)];
            rowIndex = topLeft(2);
            columnIndex = topLeft(1);
            
            % If generating a stabilized video, center all frames around
            % the first strip
           if isfield(parametersStructure, 'stabilizeVideo') && ...
                    parametersStructure.stabilizeVideo && ~foundCenter

                foundCenter = true;
                rowDifference = round(size(refFrame, 1) / 2) - rowIndex ...
                    - round(frameHeight/2);
                columnDifference = round(size(refFrame, 2) / 2) - columnIndex...
                    - round(width/2);
           end
            
            % move strip to proper position
            rowIndex = rowIndex + ((stripNumber-1) * parametersStructure.newStripHeight);
            
            % get max row/column of the strip
            maxRow = rowIndex + parametersStructure.newStripHeight - 1;
            maxColumn = columnIndex + width - 1;
            
            % transfer values of the strip pixels to the reference frame, and
            % increment the corresponding location on the counter array
            templateSelectRow = rowIndex:maxRow;
            templateSelectColumn = columnIndex:maxColumn;
            vidStart = ((stripNumber-1)*parametersStructure.newStripHeight)+1;
            vidEnd = stripNumber * parametersStructure.newStripHeight;
            columnEnd = size(videoFrame, 2);
            
            % If the strip extends beyond the frame (i.e., the frame has a
            % height of 512 pixels and strip of height 10 begins at row 511)
            % set the max row of that strip to be the last row of the frame.
            % Also make templateSelectRow smaller to match the dimensions of
            % the new strip
            if vidEnd > size(videoFrame, 1)
                difference = vidEnd - size(videoFrame, 1);
                vidEnd = size(videoFrame, 1);
                templateSelectRow = rowIndex:(maxRow-difference);
                endOfFrame = true;
            end
            
            if strip == size(frameStripsWithoutNaN, 1) 
                endOfFrame = true;
            end
            
            templateSelectRow = round(templateSelectRow);
            templateSelectColumn = round(templateSelectColumn);
            vidStart = round(vidStart);
            vidEnd = round(vidEnd);
            
            % If stabilization is enabled, center the frames in the
            % template frame
           if isfield(parametersStructure, 'stabilizeVideo') && ...
                    parametersStructure.stabilizeVideo
               
                templateSelectRow = templateSelectRow + rowDifference;
                templateSelectColumn = templateSelectColumn + columnDifference;
                
                % Handle cases in which the stabilized frames exceed the
                % dimensions of the template frame
                if max(templateSelectRow) > size(refFrame, 1)
                    stabilizeEnd = size(refFrame,1);
                    difference = max(templateSelectRow) - stabilizeEnd;
                    templateSelectRow = templateSelectRow(1):stabilizeEnd;
                    vidEnd = vidEnd - difference;
                end
                
                if max(templateSelectColumn) > size(refFrame, 2)
                    stabilizeEnd = size(refFrame, 2);
                    difference = max(templateSelectColumn) - stabilizeEnd;
                    templateSelectColumn = templateSelectColumn(1):stabilizeEnd;
                    columnEnd = columnEnd - difference;
                end
           end
           
            refFrame(templateSelectRow, templateSelectColumn) = refFrame(...
                templateSelectRow, templateSelectColumn) + double(videoFrame(...
                vidStart:vidEnd, 1:columnEnd));
            counterArray(templateSelectRow, templateSelectColumn) = counterArray...
                (templateSelectRow, templateSelectColumn) + 1;

            % Generate a stabilized video if user enables this option
            if isfield(parametersStructure, 'stabilizeVideo') && ...
                    parametersStructure.stabilizeVideo && endOfFrame
                
                stabilizedFrame = refFrame./counterArray;
                
                % Convert all NaNs to zeros
                stabilizedFrame(isnan(stabilizedFrame)) = 0;
                
                % Fill in empty strips in the video with random noise
                columnSum = sum(stabilizedFrame);
                rowSum = sum(stabilizedFrame, 2);
                
                minColumn = find(columnSum, 1, 'first');
                maxColumn = find(columnSum, 1, 'last');
                
                minRow = find(rowSum, 1, 'first');
                maxRow = find(rowSum, 1, 'last');
                
                % Replace black regions within the stabilized frames with
                % random noise
                relevantPixels = stabilizedFrame(minRow:maxRow, minColumn:maxColumn);
                indices = relevantPixels == 0;
                relevantPixels(indices) = mean(relevantPixels(~indices))...
                    + (std(relevantPixels(~indices)) * randn(sum(sum(indices)), 1));
                stabilizedFrame(minRow:maxRow, minColumn:maxColumn) = relevantPixels;
                
                % Sometimes random noise will add pixels not within 0 and 1
                stabilizedFrame(stabilizedFrame<0) = 0;
                stabilizedFrame(stabilizedFrame>1) = 1;
                
                % crop the black boundaries 
                hw = round([frameHeight width]*stabilizedVideoSizeMultiplier);
                cropRect = [round((size(stabilizedFrame)-hw)/2) hw];
                frameToWrite = imcrop(stabilizedFrame,cropRect);
                
                % Write to a video file
                writeVideo(stabilizedVideo, frameToWrite);
                
                % Reset refFrame and counterArray for the next frame.
                refFrame = zeros(size(refFrame, 1), size(refFrame, 2));
                counterArray = zeros(size(counterArray, 1), size(counterArray, 2));
                break
            end
        end
    end
end

% If a stabilized video was generated, stop the function here
if isfield(parametersStructure, 'stabilizeVideo') && parametersStructure.stabilizeVideo 
    close(stabilizedVideo)
    return
end

% divide each pixel in refFrame by the number of strips that contain that pixel
refFrame = refFrame./counterArray;

%% Take care of miscellaneous issues from preallocation and division by 0.
refFrame = Crop(refFrame);
refFrame = uint8(refFrame);

%% Save and display the reference frame.
% If this is the last iteration of FineRef, add random noise to the black
% regions to avoid getting false peaks in the final StripAnalysis.
if isfield(parametersStructure, 'addNoise') && parametersStructure.addNoise == true
    % Replace remaining black regions with random noise
    indices = refFrame == 0;
    refFrame(indices) = mean(refFrame(~indices)) + (std(double(refFrame(~indices))) ...
        * randn(sum(sum(indices)), 1));
end

fileName(end-3:end) = [];
fileName(end+1:end+9) = '_refframe';
save(fileName, 'refFrame');
if ~isfield(parametersStructure, 'axesHandles')
    % Show only if not using GUI.
    figure('Name', 'Reference Frame')
    imshow(refFrame);
end
end