function outputVideo = RemoveStimuli(inputVideo, stimulus, parametersStructure, removalAreaSize)
%REMOVE STIMULI Finds and removes stimuli from each frame. 
% Stimulus locations are saved in a mat file, and the video with stimuli
% removed is saved with "_nostim" suffix.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is either the path to the video, or the video matrix itself. 
%   In the former situation, the result is stored in a new video file with 
%   '_nostim' appended to the input video file name. In the latter situation, 
%   no video is written and the result is returned. In either case, a mat 
%   file with the stimulus locations are saved, as well as the mean and 
%   standard deviation of the pixels in each frame. This mat file has 
%   '_stimlocs' appeneded to the input video file name, or is simply called
%   '.stimlocs.mat' if a video matrix was passed in (hidden file).
%
%   |stimulus| is a path to a stimulus or a struct containing a |size|
%   field which is the size of the stimulus in pixels (default 11), and a
%   |thickness| field which is the thickness of the default cross shape in
%   pixels (default 1). (default is dynamically generated stimulus with
%   aforementioned defaults)
%
%   |parametersStructure| is a struct as specified below.
%
%   |removalAreaSize| is the size of the rectangle to remove from the
%   video, centered around the identified stimulus location. The format is
%   [width length], given in pixels. (default [11 11])
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite          : set to true to overwrite existing files.
%                        Set to false to abort the function call if the
%                        files already exist. (default false)
%   enableVerbosity    : set to true to report back plots during execution.
%                        (default false)
%   axesHandles        : axes handle for giving feedback. if not
%                        provided or empty, new figures are created.
%                        (relevant only when enableVerbosity is true)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       parametersStructure.enableVerbosity = true;
%       parametersStructure.overwrite = true;
%       stimulus = struct;
%       stimulus.size = 11;
%       stimulus.thickness = 1;
%       removalAreaSize = [11 11];
%       RemoveStimuli(inputVideo, stimulus, parametersStructure, ...
%                             removalAreaSize);

%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
    matFileName = '.stimlocs.mat';
end

%% Handle overwrite scenarios.

if writeResult
    outputVideoPath = [inputVideo(1:end-4) '_nostim' inputVideo(end-3:end)];
    matFileName = [inputVideo(1:end-4) '_stimlocs'];
    if ~exist([matFileName '.mat'], 'file') && ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasWarning('RemoveStimuli() did not execute because it would overwrite existing file.', parametersStructure);
        return;
    else
        RevasWarning('RemoveStimuli() is proceeding and overwriting an existing file.', parametersStructure);
    end
end

%% Convert stimulus to matrix
% Two stimulus input types are acceptable:
% - Path to an image of the stimulus
% - A struct describing |size| and |thickness| of stimulus

if ischar(stimulus)
    % Read image from the path
    stimulus = imread(stimulus);
elseif isstruct(stimulus)
    % Both size and thickness must be odd
    if ~mod(stimulus.size, 2) == 1 || ~mod(stimulus.thickness, 2) == 1
        error('stimulus.size and stimulus.thickness must be odd');
    elseif stimulus.size < stimulus.thickness
        error('stimulus.size must be greater than stimulus.thickness');
    end
    % Draw the stimulus cross based on struct parameters provided.
    stimulusMatrix = ones(stimulus.size);
    stimulusMatrix(1:(stimulus.size-stimulus.thickness)/2, ...
        1:(stimulus.size-stimulus.thickness)/2) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end, ...
        1:(stimulus.size-stimulus.thickness)/2) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(1:(stimulus.size-stimulus.thickness)/2, ...
        stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end, ...
        stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulus = stimulusMatrix;
end

%% Set parameters to defaults if not specified.

% Validation on the input stimulus was already performed as it was converted to a
% matrix.

if ~isfield(parametersStructure, 'enableVerbosity')
    parametersStructure.enableVerbosity = false;
end

if nargin == 3
   stimulusSize = size(stimulus);
else
   stimulusSize = removalAreaSize;
   if size(stimulusSize) == [2 1]
       stimulusSize = stimulusSize';
   elseif size(stimulusSize) ~= [1 2]
     error('stimulusSize must be a 1 by 2 size array'); 
   elseif ~IsNaturalNumber(stimulusSize(1)) || ~IsNaturalNumber(stimulusSize(2))
      error('values of stimulusSize must be natural numbers'); 
   end
end

if ~writeResult && ~isfield(parametersStructure, 'FrameRate')
    parametersStructure.FrameRate = 30;
    RevasWarning('using default parameter for FrameRate', parametersStructure);
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Find stimulus location of each frame

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    % Determine dimensions of video.
    reader = VideoReader(inputVideo);
    samplingRate = reader.parametersStructure.FrameRate;
    width = reader.Width;
    height = reader.Height;
    numberOfFrames = reader.parametersStructure.FrameRate * reader.Duration;
    
else
    samplingRate = parametersStructure.FrameRate;
    
    % Determine dimensions of video.
    [height, width, numberOfFrames] = size(inputVideo);
end

% Populate time array
timeArray = (1:numberOfFrames)' / samplingRate;   

% Read and call normxcorr2() to find stimulus frame by frame.
% Note that calculation for each array value does not end with this loop,
% the logic below the loop in this section perform remaining operations on
% the values but are done outside of the loop in order to take advantage of
% vectorization (that is, if verbosity is not enabled since if it was, then
% these operations must be computed immediately so that the correct
% stimulus location values can be plotted as early as possible).

% preallocate two columns for horizontal and vertical movements
stimulusLocationInEachFrame = NaN(numberOfFrames, 2);
% preallocate mean and standarddeviation result vectors
meanOfEachFrame = NaN(numberOfFrames, 1);
standardDeviationOfEachFrame = NaN(numberOfFrames, 1);

% Threshold value for detecting stimulus. Any peak value below this
% threshold will not be marked as a stimulus.
stimulusThresholdValue = 0.8; % TODO hard-coded threshold.

for frameNumber = 1:numberOfFrames
    if ~abortTriggered
        if writeResult
            frame = readFrame(reader);
            if ndims(frame) == 3
                frame = rgb2gray(frame);
            end
        else
            frame = inputVideo(1:end, 1:end, frameNumber);
        end

        correlationMap = normxcorr2(stimulus, frame);

        findPeakParametersStructure.enableGaussianFiltering = false;
        findPeakParametersStructure.stripHeight = height;    
        [xPeak, yPeak, peakValue, ~] = ...
            FindPeak(correlationMap, findPeakParametersStructure);
        clear findPeakParametersStructure;

        % Show surface plot for this correlation if verbosity enabled
        if parametersStructure.enableVerbosity
            if isfield(parametersStructure, 'axesHandles')
                axes(parametersStructure.axesHandles(1));
                colormap(parametersStructure.axesHandles(1), 'default');
            else
                figure(1);
            end
            [surfX,surfY] = meshgrid(1:size(correlationMap,2), 1:size(correlationMap,1));
            surf(surfX, surfY, correlationMap,'linestyle','none');
            title([num2str(frameNumber) ' out of ' num2str(numberOfFrames)]);
            xlim([1 size(correlationMap,2)]);
            ylim([1 size(correlationMap,1)]);
            zlim([-1 1]);

            % Mark the identified peak on the plot with an arrow.
            text(xPeak, yPeak, peakValue, '\downarrow', 'Color', 'red', ...
                'FontSize', 20, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

            drawnow;
        end

        if peakValue < stimulusThresholdValue
            continue;
        end

        stimulusLocationInEachFrame(frameNumber,:) = [xPeak yPeak];
        
        % Adjust by stimulus size if necessary.
        if nargin ~= 3
           stimulusLocationInEachFrame(frameNumber,1) = ...
               stimulusLocationInEachFrame(frameNumber,1) + floor((removalAreaSize(2) - size(stimulus, 2)) / 2);
           stimulusLocationInEachFrame(frameNumber,2) = ...
               stimulusLocationInEachFrame(frameNumber,2) + floor((removalAreaSize(1) - size(stimulus, 1)) / 2);
        end

        % If verbosity is enabled, also show stimulus location plot with points
        % being plotted as they become available.
        if parametersStructure.enableVerbosity

            % Plotting bottom right corner of box surrounding stimulus.
            if isfield(parametersStructure, 'axesHandles')
                axes(parametersStructure.axesHandles(2));
                colormap(parametersStructure.axesHandles(2), 'default');
            else
                figure(2);
            end
            plot(timeArray, stimulusLocationInEachFrame);
            title('Stimulus Locations');
            xlabel('Time (sec)');
            ylabel('Stimulus Locations (pixels)');
            legend('show');
            legend('Horizontal Location', 'Vertical Location');
        end
        
        % Find mean and standard deviation of each frame.
        meanOfEachFrame(frameNumber) = mean2(frame);
        standardDeviationOfEachFrame(frameNumber) = std2(frame);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Remove target / replace it with noise here
        location = stimulusLocationInEachFrame(frameNumber,:);
        if isnan(location)
            continue;
        end

        % Account for removal target at edge of array
        xLow = location(2)-stimulusSize(1)+1;
        xHigh = location(2);
        yLow = location(1)-stimulusSize(2)+1;
        yHigh = location(1);

        targetArea = frame(max(xLow, 1) :  min(xHigh, height),max(yLow, 1) : min(yHigh, width));
        targetArea(targetArea > 250) = 0;
        toBeRemoved = imbinarize(targetArea,0.15) == 0;
        
        % Generate noise
        % (this gives noise with mean = 0, sd = 1)
        noise = randn(sum(toBeRemoved(:)),1);

        % Adjust to the mean and sd of current frame
        noise = noise * standardDeviationOfEachFrame(frameNumber) + 1.2*meanOfEachFrame(frameNumber);
        targetArea(toBeRemoved) = noise;
        
        frame(max(xLow, 1) : min(xHigh, height),max(yLow, 1) : min(yHigh, width)) = targetArea;
        
        if writeResult
            writeVideo(writer, frame);
        else
           inputVideo(1:end, 1:end, frameNumber) = frame; 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end

if writeResult
    close(writer);
else
    outputVideo = inputVideo;
end

%% Save to output mat file
save(matFileName, 'stimulusLocationInEachFrame', 'stimulusSize', ...
    'meanOfEachFrame', 'standardDeviationOfEachFrame');

end
