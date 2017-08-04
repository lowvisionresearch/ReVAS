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
%   params must have the fields: params.scalingFactor, params.refFrameNumber, 
%   an optional parameter that designates which frame to use as the initial 
%   scaled down reference frame, params.overwrite (optional), 
%   params.enableVerbosity, which is either 0, 1, or 2. Verbosity of 0 will 
%   only save the output in a MatLab file. Verbosity of 1 will display the 
%   final result. Verbosity of 2 will show the progress of the program. 
%   scalingFactor is the factor by which each frame will be multiplied to 
%   get the approximate frame shifts.
%   params also needs params.peakRatio and params.minimumPeakThreshold
%
%   Example: 
%       videoPath = 'MyVid.avi';
%       params.enableGPU = false;
%       params.overwrite = true;
%       params.refFrameNumber = 15;
%       params.enableVerbosity = 2;
%       params.scalingFactor = 0.5;
%       CoarseRef(params, filename);

%% Handle miscellaneous preliminary info

% Identify which frames are bad frames
nameEnd = strfind(videoPath,'dwt_');
blinkFramesPath = [videoPath(1:nameEnd+length('dwt_')-1) 'blinkframes'];
try
    load(blinkFramesPath, 'badFrames');
catch 
end

% Check to see if operations can be performed on GPU and whether the
% user wants to do so if there is a GPU
enableGPU = (gpuDeviceCount > 0) & parametersStructure.enableGPU;

% Write the output to a new MatLab file. First remove the '.avi' extension
outputFileName = videoPath;
outputFileName((end-3):end) = [];

% Extend name because the file has been processed by coarseref
outputFileName(end + 1: end + 10) = '_coarseref';

% Handle overwrite scenarios.
if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['CoarseRef() did not execute because it would overwrite existing file. (' outputFileName ')'], parametersStructure);
    coarseRefFrame = [];
    return;
else
    RevasWarning(['CoarseRef() is proceeding and overwriting an existing file. (' outputFileName ')'], parametersStructure);  
end

%% Initialize variables
% get video info
v = VideoReader(videoPath);
timeToRemember = v.CurrentTime;
frameRate = v.FrameRate;
totalFrames = v.frameRate * v.Duration;
tinyVideoName = videoPath;
tinyVideoName(end-3:end) = [];
tinyVideoName(end+1:end+11) = '_shrunk.avi';
tinyVideo = VideoWriter(tinyVideoName);

% if no frame number is designated as the original reference frame, then
% the default frame should be the "middle" frame of the total frames (i.e.,
% if frameRate = 30 Hz and duration is 2 seconds, there are 60 total frames
% and the default frame should therefore be the 30th frame).
if ~isfield(parametersStructure, 'refFrameNumber')
    parametersStructure.refFrameNumber = totalFrames/2;
end
if exist('badFrames', 'var')
    while any(badFrames == parametersStructure.refFrameNumber)
        if parametersStructure.refFrameNumber ~= 1
            parametersStructure.refFrameNumber = parametersStructure.refFrameNumber...
                - 1;
        else
            parametersStructure.refFrameNumber = parametersStructure.refFrameNumber...
                + 1;
        end
    end
end
%% Create new shrunken video and call strip analysis on it
open(tinyVideo)

% Shrink each frame and write to a new video so that stripAnalysis can be
% called, using each frame as one "strip"
while hasFrame(v)
    currFrame = readFrame(v);
    shrunkFrame = imresize(currFrame, parametersStructure.scalingFactor);
    writeVideo(tinyVideo, shrunkFrame)
end

close(tinyVideo)

tinyVideo = VideoReader(tinyVideoName);
frameNumber = 1;

while hasFrame(tinyVideo)
    currFrame = readFrame(tinyVideo);
    if frameNumber == parametersStructure.refFrameNumber
        currFrame = rgb2gray(currFrame);
        temporaryRefFrame = double(currFrame)/255;
        break
    end
    frameNumber = frameNumber + 1;
end

params = parametersStructure;
params.stripHeight = size(currFrame, 1);
params.samplingRate = frameRate;
params.stripWidth = size(currFrame, 2);
[~, usefulEyePositionTraces, ~, ~] = StripAnalysis(tinyVideoName, ...
    temporaryRefFrame, params);

% Scale the coordinates back up. Throw out information from bad frames
framePositions = zeros(totalFrames, 2);
for row = 1:size(usefulEyePositionTraces, 1)
    if exist('badFrames', 'var') && any(badFrames==row)
        framePositions(row, :) = NaN;
    else
        framePositions(row, 1) = usefulEyePositionTraces(row, 1) * ...
            1/parametersStructure.scalingFactor;
        framePositions(row, 2) = usefulEyePositionTraces(row, 2) * ...
            1/parametersStructure.scalingFactor;
    end
end

% %% Perform cross-correlation using the temporary reference frame
% frameNumber = 1;
% while frameNumber <= totalFrames
%     startIndex = ((frameNumber-1) * columns) + 1;
%     endIndex = frameNumber*columns;
%     
%     % Perform cross-correlation on GPU if enabled
%     if enableGPU
%         currFrame = gpuArray(frames(:, startIndex:endIndex));
%         temporaryRefFrame = gpuArray(temporaryRefFrame);
%     else
%         currFrame = frames(:, startIndex:endIndex);
%     end
%     
%     c = normxcorr2(temporaryRefFrame, currFrame);
% 
%     % Find peak in cross-correlation using FindPeak function. FindPeak
%     % takes in a parametersStructure that has fields for Gaussian filtering
%     % and standard deviation.
%     parametersStructure.enableGaussianFiltering = 0;
%     [xPeak, yPeak, peakValue, secondPeakValue] = FindPeak(c, parametersStructure);
%     
%     % Account for the padding that normxcorr2 adds.
%     if enableGPU
%         yoffSet = gather(yPeak-size(currFrame,1));
%         xoffSet = gather(xPeak-size(currFrame,2));
%         xPeak = gather(xPeak);
%         yPeak = gather(yPeak);
%         peakValue = gather(peakValue);
%     else
%         yoffSet = yPeak-size(currFrame,1);
%         xoffSet = xPeak-size(currFrame,2);
%     end
%     
%     % Use the peak to get the top-left coordinate of the frame relative 
%     % to the temporary reference frame.
%     peakRatio = secondPeakValue/peakValue;
%     if peakRatio >= params.peakRatio && peakValue >= params.minimumPeakThreshold...
%             && peakRatio < 0.99
%         columnCoordinate = xoffSet + 1;
%         rowCoordinate = yoffSet + 1;
%    
%         % Scale back up. Currently the program is using the shrunken frames
%         % as a rough reference. Therefore, the positions need to be
%         % multiplied by the reciprocal of the scaling factor to get the
%         % real frame movements (i.e., if a frame is shrunken by 1/2 and
%         % moves 6 units to the right of the shrunken reference frame, then 
%         % the real frame moved 12 units to the right relative to the real 
%         % reference frame).
%         framePositions(frameNumber, 1) = (1/parametersStructure.scalingFactor) ...
%             * rowCoordinate;
%         framePositions(frameNumber, 2) = (1/parametersStructure.scalingFactor) ...
%             * columnCoordinate;
%     else
%         framePositions(frameNumber, :) = NaN;
%     end
% 
%     % Show surface plot for this correlation if verbosity enabled
%     if parametersStructure.enableVerbosity == 2
%         if isfield(parametersStructure, 'axesHandles')
%             axes(parametersStructure.axesHandles(1));
%         else
%             figure(1);
%         end
%         [surfX,surfY] = meshgrid(1:size(c,2), 1:size(c,1));
%         surf(surfX, surfY, c,'linestyle','none');
%         title([num2str(frameNumber) ' out of ' num2str(totalFrames)]);
%         xlim([1 size(c,2)]);
%         ylim([1 size(c,1)]);
%         zlim([-1 1]);
%         
%         % Mark the identified peak on the plot with an arrow.
%         text(xPeak, yPeak, peakValue, '\downarrow', 'Color', 'red', ...
%             'FontSize', 20, 'HorizontalAlignment', 'center', ...
%             'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
%         
%         drawnow;  
%         
%         % Also plot the positions of the frames as time progresses
%         timeAxis = (1/v.frameRate):(1/v.frameRate):(frameNumber/v.frameRate);
%         if isfield(parametersStructure, 'axesHandles')
%             axes(parametersStructure.axesHandles(2));
%         else
%             figure(2);
%         end
%         plot(timeAxis, framePositions(1:frameNumber, :));
%         title('Coarse Frame Shifts (scaled up)');
%         xlabel('Time (sec)');
%         ylabel('Approximate Frame Shifts (pixels)');
%         legend('show');
%         legend('Vertical Traces', 'Horizontal Traces');
%         
%         % Adjust margins to guarantee that the user can see all points
%         xlim([0 max(timeAxis)*1.1])
%         mostNegative = min(min(framePositions(:, 1)), min(framePositions(:, 2)));
%         mostPositive = max(max(framePositions(:, 1)), max(framePositions(:, 2)));
%         if isnan(mostNegative)
%             mostNegative = -200;
%         end
%         if isnan(mostPositive)
%             mostPositive = 1;
%         end
%             
%         ylim([mostNegative*1.1 mostPositive*1.1])
%       
%     end
% 
%     frameNumber = frameNumber + 1;
%     
% end

%% Remove NaNs in framePositions
% First handle the case in which NaNs are at the beginning of
% framePositions
numberOfNaNs = 0;
NaNIndices = find(isnan(framePositions));
if ~isempty(NaNIndices)
    if NaNIndices(1) == 1
        numberOfNaNs = 1;
        index = 1;
        while index <= size(NaNIndices, 1)
            if NaNIndices(index + 1) == NaNIndices(index) + 1
                numberOfNaNs = numberOfNaNs + 1;
            else
                break
            end
            index = index + 1;
        end
        framePositions(1:numberOfNaNs, :) = [];
    end
end

% Then replace the rest of the NaNs with linear interpolation, done
% manually in a helper function. NaNs at the end of framePositions will be
% deleted.
[filteredStripIndices1, ~] = FilterStrips(framePositions(:, 1));
[filteredStripIndices2, ~] = FilterStrips(framePositions(:, 2));
framePositions = [filteredStripIndices1 filteredStripIndices2];

save('framePositions', 'framePositions');

%% Set up the counter array and the template for the coarse reference frame.
height = size(currFrame, 1) * 1/parametersStructure.scalingFactor;
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

% "Rewind" the video so we can add to the template for the coarse
% reference frame
v.CurrentTime = timeToRemember;

if enableGPU
    totalFrames = gpuArray(totalFrames);
    framePositions = gpuArray(framePositions);
    counterArray = gpuArray(counterArray);
    coarseRefFrame = gpuArray(coarseRefFrame);
end

for frameNumber = 1:totalFrames
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