function RemoveStimuli(inputVideoPath, parametersStructure)
%REMOVE STIMULI Records in a mat file the location of the stimulus
%in each frame of the video.
%   The result is stored with '_nostim' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputVideoPath = [inputVideoPath(1:end-4) '_nostim' inputVideoPath(end-3:end)];
matFileName = [inputVideoPath(1:end-4) '_stimlocs'];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['RemoveStimuli() did not execute because it would overwrite existing file. (' outputVideoPath ')']);
    return;
else
    RevasWarning(['RemoveStimuli() is proceeding and overwriting an existing file. (' outputVideoPath ')']);
end

%% Load mat file with output from |FindStimulusLocations|

load(matFileName);

% Variables that should be Loaded now:
% - stimulusLocationInEachFrame
% - stimulusSize
% - meanOfEachFrame
% - standardDeviationOfEachFrame

%% Remove stimuli frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

numberOfFrames = size(videoInputArray, 3);

for frameNumber = 1:numberOfFrames
    
    % Generate noise
    % (this gives noise with mean = 0, sd = 1)
    noise = randn(stimulusSize);
    
    % Adjust to the mean and sd of current frame
    noise = ...
        noise * standardDeviationOfEachFrame(frameNumber) + meanOfEachFrame(frameNumber);
    
    location = stimulusLocationInEachFrame(frameNumber,:);
    
    % Account for removal target at edge of array
    xLow = location(2)-stimulusSize(1)+1;
    xHigh = location(2);
    yLow = location(1)-stimulusSize(2)+1;
    yHigh = location(1);
    
    xDiff = 0;
    yDiff = 0;
    
    if xLow < 1
        xDiff = -xLow+1;
    elseif xHigh > size(videoInputArray, 2)
        xDiff = xHigh - size(videoInputArray, 2);
    end
    
    if yLow < 1
        yDiff = -yLow+1;
    elseif yHigh > size(videoInputArray, 1)
        yDiff = yHigh - size(videoInputArray, 1);
    end
    videoInputArray(max(location(2)-stimulusSize(1)+1, 1) : ...
        min(location(2), size(videoInputArray, 2)),...
        max(location(1)-stimulusSize(2)+1, 1) : ...
        min(location(1), size(videoInputArray, 1)), ...
        frameNumber) = ...
        noise(1:end-xDiff, 1:end-yDiff);
end

writeVideo(writer, videoInputArray);
close(writer);

end

