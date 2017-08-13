function [refFrame] = MakeMontage(params, fileName)
% MakeMontage    Reference frame.
%   MakeMontage(params, fileName) generates a reference frame by averaging
%   all the pixel values across all frames in a video.
%   
%   Note: The params variable should have the attributes params.stripHeight,
%   params.positions, and params.time.
%   params.newStripHeight is an optional parameter.
%
%   Example: 
%       params.stripHeight = 15;
%       params.time = 1/540:1/540:1;
%       params.positions = randn(540, 2);
%       MakeMontage(params,'fileName');

%% Prepare a new field for newStripHeight
% newStripHeight will give the requisite information for interpolation
% between the spaced-out strips later.
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

%% Identify which frames are bad frames
nameEnd = strfind(fileName,'dwt_');
blinkFramesPath = [fileName(1:nameEnd+length('dwt_')-1) 'blinkframes'];
try
    load(blinkFramesPath, 'badFrames');
catch
    badFrames = [];
end

%% Set up all variables
stripIndices = params.positions;
t1 = params.time;

% grabbing info about the video and strips
videoInfo = VideoReader(fileName);
frameHeight = videoInfo.Height;
width = videoInfo.Width;
frameRate = videoInfo.Framerate;
totalFrames = frameRate * videoInfo.Duration;
stripsPerFrame = floor(frameHeight/params.newStripHeight);

% setting up templates for reference frame and counter array
counterArray = zeros(frameHeight*3);
refFrame = zeros(frameHeight*3);

% Negate all positions. stripIndices tells how far a strip has moved from
% the reference--therefore, to compensate for that movement, the
% stripIndices have to be negated when placing the strip on the template
% frame.
stripIndices = -stripIndices;

%% Set up the interpolation
% scaling the time array to accomodate new strip height
scalingFactor = ((params.stripHeight)/2)/(frameRate*frameHeight);
t1 = t1 + scalingFactor;
dt = params.newStripHeight / (frameRate * frameHeight);
t2 = dt:dt:videoInfo.Duration + scalingFactor;

% Make sure that both time arrays have the same dimensions
if size(t2, 1) ~= size(t1, 1) && size(t2, 2) ~= size(t1, 2)
    t2 = t2';
end

%% Remove NaNs in stripIndices

% Then replace the rest of the NaNs with linear interpolation, done
% manually in a helper function. NaNs at the end of stripIndices will be
% deleted, along with their corresponding time points.
[filteredStripIndices1, lengthCutOut1, numberOfNaNs1] = FilterStrips(stripIndices(:, 1));
[filteredStripIndices2, lengthCutOut2, numberOfNaNs2] = FilterStrips(stripIndices(:, 2));
numberOfNaNs = max(numberOfNaNs1, numberOfNaNs2);
lengthCutOut = max(lengthCutOut1, lengthCutOut2);

% Handle the case in which the two column vectors are different sizes
difference1 = numberOfNaNs1 - numberOfNaNs2;
difference2 = lengthCutOut1 - lengthCutOut2;

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
if numberOfNaNs >= 1
    t1(1:numberOfNaNs) = [];
end

if lengthCutOut >= 1
    t1(end-lengthCutOut+1:end) = [];
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
for stripNumber = numberOfNaNs + 1:size(interpolatedPositions, 1) + numberOfNaNs
    interpolatedPositions(stripNumber-numberOfNaNs, 3) = stripNumber;
end

% Remove leftover NaNs. Need to use while statement because the size of
% interpolatedPositions changes each time I remove an NaN
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

% Scale the strip coordinates so that all values are positive. Take the
% negative value with the highest magnitude in each column of
% framePositions and add that value to all the values in the column. This
% will zero the most negative value (like taring a weight scale). Then add
% a positive integer to make sure all values are > 0 (i.e., if we leave the
% zero'd value in framePositions, there will be indexing problems later,
% since MatLab indexing starts from 1 not 0).
column1 = interpolatedPositions(:, 1);
column2 = interpolatedPositions(:, 2); 
if column1(column1<0)
   mostNegative = max(-1*column1);
   interpolatedPositions(:, 1) = interpolatedPositions(:, 1) + mostNegative + 2;
end

if column2(column2<0)
    mostNegative = max(-1*column2);
    interpolatedPositions(:, 2) = interpolatedPositions(:, 2) + mostNegative + 2;
end

if column1(column1<0.5)
    interpolatedPositions(:, 1) = interpolatedPositions(:, 1) + 2;
end

if column2(column2<0.5)
    interpolatedPositions(:, 2) = interpolatedPositions(:, 2) + 2;
end

% Round the final scaled positions to get valid matrix indices
interpolatedPositions = round(interpolatedPositions);

% Leftover and skips will be used to correct for rounding errors; see the 
% for-loop below for details 
leftover = (frameHeight/params.newStripHeight) - floor(frameHeight/params.newStripHeight);
leftoverCopy = leftover;
skips = 0;

%% Use interpolatedPositions to generate the reference frame.
for frameNumber = 1:totalFrames
    % By default, the current frame is not one that needs the correction
    % factor for rounding
    correctionFrame = false;
    
    % Read frame and convert pixel values to signed integers
    videoFrame = double(readFrame(videoInfo))/255;
    
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
        if startFrameStrips <= interpolatedPositions(k, 3) && ...
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
        continue
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

            % Keep track of the stripNumber so we can shift it accordingly
            stripNumber = frameStripsWithoutNaN(strip, 4);

            % row and column "coordinates" of the top left pixel of each strip
            topLeft = [frameStripsWithoutNaN(strip, 1), frameStripsWithoutNaN(strip, 2)];
            rowIndex = topLeft(2);
            columnIndex = topLeft(1);

            % move strip to proper position
            rowIndex = rowIndex + ((stripNumber-1) * params.newStripHeight);

            % get max row/column of the strip
            maxRow = rowIndex + params.newStripHeight - 1;
            maxColumn = columnIndex + width - 1;

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

            templateSelectRow = round(templateSelectRow);
            templateSelectColumn = round(templateSelectColumn);
            vidStart = round(vidStart);
            vidEnd = round(vidEnd);

            refFrame(templateSelectRow, templateSelectColumn) = refFrame(...
                templateSelectRow, templateSelectColumn) + videoFrame(...
                vidStart:vidEnd, :);
            counterArray(templateSelectRow, templateSelectColumn) = counterArray...
                (templateSelectRow, templateSelectColumn) + 1;
        end
    end
end


% divide each pixel in refFrame by the number of strips that contain that pixel
refFrame = refFrame./counterArray;

%% Take care of miscellaneous issues from preallocation and division by 0.

% Convert any NaN values in the reference frame to a 0. Otherwise, running
% strip analysis on this new frame will not work
NaNindices = find(isnan(refFrame));
for k = 1:size(NaNindices)
    NaNindex = NaNindices(k);
    refFrame(NaNindex) = 0;
end

% Crop out the leftover 0 padding from the original template. First check
% for 0 rows
k = 1;
while k<=size(refFrame, 1)
    if refFrame(k, :) == 0
        refFrame(k, :) = [];
        continue
    end
    k = k + 1;
end

% Then sweep for 0 columns
k = 1;
while k<=size(refFrame, 2)
    if refFrame(:, k) == 0
        refFrame(:, k) = [];
        continue
    end
    k = k + 1;
end
%% Save and display the reference frame.
fileName(end-3:end) = [];
fileName(end+1:end+9) = '_refframe';
save(fileName, 'refFrame');
figure('Name', 'Reference Frame')
imshow(refFrame);

end