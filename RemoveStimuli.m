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
    warning('RemoveStimuli() did not execute because it would overwrite existing file.');
    return;
else
    warning('RemoveStimuli() is proceeding and overwriting an existing file.');
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
    videoInputArray(location(2)-stimulusSize(1)+1 : location(2), ...
        location(1)-stimulusSize(2)+1 : location(1), frameNumber) = noise;
    
end

writeVideo(writer, videoInputArray);
close(writer);

end

