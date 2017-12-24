function RemoveStimuli(inputVideoPath, parametersStructure)
%REMOVE STIMULI Removes stimulus from each frame.
%   Removes stimulus from each frame, according to the stimuli positions
%   given by |FindStimulusLocations|. Fills the space with noise of similar
%   mean and standard deviation as the rest of the frame.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video. The result is stored with 
%   '_nostim' appended to the input video file name.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite      : set to true to overwrite existing files.
%                    Set to false to abort the function call if the
%                    files already exist. (default false)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = true;
%       RemoveStimuli(inputVideoPath, parametersStructure);

%% Handle overwrite scenarios.
outputVideoPath = [inputVideoPath(1:end-4) '_nostim' inputVideoPath(end-3:end)];
matFileName = [inputVideoPath(1:end-4) '_stimlocs'];
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['RemoveStimuli() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
    return;
else
    RevasWarning(['RemoveStimuli() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
end

%% Set parameters to defaults if not specified.

% There are no required parameters that need to be set to default values.

%% Load mat file with output from |FindStimulusLocations|

load(matFileName);

% Variables that should be Loaded now:
% - stimulusLocationInEachFrame
% - stimulusSize
% - meanOfEachFrame
% - standardDeviationOfEachFrame

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Remove stimuli frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

% Determine dimensions of video.
reader = VideoReader(inputVideoPath);
numberOfFrames = reader.NumberOfFrames;
width = reader.Width;
height = reader.Height;

% Remake this variable since readFrame() cannot be called after
% NumberOfFrames property is accessed.
reader = VideoReader(inputVideoPath);

% Read, remove stimuli, and write frame by frame.
for frameNumber = 1:numberOfFrames
    if ~abortTriggered
        frame = readFrame(reader);
        
        % Generate noise
        % (this gives noise with mean = 0, sd = 1)
        noise = randn(stimulusSize);

        % Adjust to the mean and sd of current frame
        noise = ...
            noise * standardDeviationOfEachFrame(frameNumber) + meanOfEachFrame(frameNumber);

        location = stimulusLocationInEachFrame(frameNumber,:);

        if isnan(location)
            continue;
        end

        % Account for removal target at edge of array
        xLow = location(2)-stimulusSize(1)+1;
        xHigh = location(2);
        yLow = location(1)-stimulusSize(2)+1;
        yHigh = location(1);

        xDiff = 0;
        yDiff = 0;

        if xLow < 1
            xDiff = -xLow+1;
        elseif xHigh > height
            xDiff = xHigh - height;
        end

        if yLow < 1
            yDiff = -yLow+1;
        elseif yHigh > width
            yDiff = yHigh - width;
        end
        frame(max(xLow, 1) : ...
            min(xHigh, height),...
            max(yLow, 1) : ...
            min(yHigh, width)) = ...
            noise(1:end-xDiff, 1:end-yDiff);
        writeVideo(writer, frame);
    end
end

close(writer);

end
