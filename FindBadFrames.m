function FindBadFrames(inputVideoPath, parametersStructure)
%REMOVE STIMULI Records in a mat file the location of the stimulus
%in each frame of the video.
%   The result is stored with '_nostim' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

stimLocsMatFileName = [inputVideoPath(1:end-4) '_stimlocs'];
badFramesMatFileName = [inputVideoPath(1:end-4) '_badframes'];

%% Handle overwrite scenarios.
if ~exist([badFramesMatFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    warning('FindBadFrames() did not execute because it would overwrite existing file.');
    return;
else
    warning('FindBadFrames() is proceeding and overwriting an existing file.');
end

%% Set thresholdValue

if ~isfield(parametersStructure, 'thresholdValue')
    thresholdValue = 2;
else
    thresholdValue = parametersStructure.thresholdValue;
end

%% Load mat file with output from |FindStimulusLocations|

load(stimLocsMatFileName);

% Variables that should be Loaded now:
% - stimulusLocationInEachFrame
% - stimulusSize
% - meanOfEachFrame
% - standardDeviationOfEachFrame

%% Identify bad frames

% Use the differences of the means to identify bad frames
meanDiffs = diff(meanOfEachFrame);
medianOfMeanDiffs = median(meanDiffs);
standardDeviationOfMeanDiffs = sqrt(median(meanDiffs.^2) - medianOfMeanDiffs^2);
meanThreshold = medianOfMeanDiffs - thresholdValue * standardDeviationOfMeanDiffs;
meanBadFrames = [0; meanDiffs]' < meanThreshold;
% Mark frames near bad frames as bad as well
meanBadFrames = conv(meanBadFrames, [1 1 1], 'same') >= 1;

% Use the differences of the standard deviations to identify bad frames
standardDeviationDiffs = diff(standardDeviationOfEachFrame);
medianOfStandardDeviationDiffs = median(standardDeviationDiffs);
standardDeviationOfStandardDeviationDiffs = ...
    sqrt(median(standardDeviationDiffs.^2) - medianOfStandardDeviationDiffs^2);
standardDeviationThreshold = ...
    medianOfStandardDeviationDiffs - thresholdValue * standardDeviationOfStandardDeviationDiffs;
standardDeviationBadFrames = [0; standardDeviationDiffs]' < standardDeviationThreshold;
% Mark frames near bad frames as bad as well
standardDeviationBadFrames = conv(standardDeviationBadFrames, [1 1 1], 'same') >= 1;

% Combine results from both mean and standard deviation approaches.
badFrames = or(meanBadFrames, standardDeviationBadFrames);
badFrames = find(badFrames);

%% Save to output mat file
save(badFramesMatFileName, 'badFrames');

end

