function FindStimulusLocations(inputVideoPath, stimulus, parametersStructure, removalAreaSize)
%FIND STIMULUS LOCATIONS Records in a mat file the location of the stimulus
%in each frame of the video.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video. The result is stored with 
%   '_stimlocs' appended to the input video file name (and the file type
%   is now .mat). The mean and standard deviation of the pixels in each
%   frame are also saved as two separate arrays in this output mat file.
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
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.enableVerbosity = true;
%       parametersStructure.overwrite = true;
%       stimulus = struct;
%       stimulus.size = 11;
%       stimulus.thickness = 1;
%       removalAreaSize = [11 11];
%       FindStimulusLocations(inputVideoPath, stimulus, parametersStructure, ...
%                             removalAreaSize);

%% Handle overwrite scenarios.
matFileName = [inputVideoPath(1:end-4) '_stimlocs'];
if ~exist([matFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning('FindStimulusLocations() did not execute because it would overwrite existing file.', parametersStructure);
    return;
else
    RevasWarning('FindStimulusLocations() is proceeding and overwriting an existing file.', parametersStructure);
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

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Find stimulus location of each frame

% Determine dimensions of video.
reader = VideoReader(inputVideoPath);
samplingRate = reader.FrameRate;
width = reader.Width;
height = reader.Height;
numberOfFrames = reader.Framerate * reader.Duration;

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
        frame = readFrame(reader);

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
    end
end

%% Save stimulus size
% Adjust by stimulus size if necessary.
if nargin ~= 3
   stimulusLocationInEachFrame(:,1) = stimulusLocationInEachFrame(:,1) + floor((removalAreaSize(2) - size(stimulus, 2)) / 2);
   stimulusLocationInEachFrame(:,2) = stimulusLocationInEachFrame(:,2) + floor((removalAreaSize(1) - size(stimulus, 1)) / 2);
end

%% Save to output mat file
save(matFileName, 'stimulusLocationInEachFrame', 'stimulusSize', ...
    'meanOfEachFrame', 'standardDeviationOfEachFrame');

end
