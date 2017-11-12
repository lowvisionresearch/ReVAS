function FindBlinkFrames(inputVideoPath, parametersStructure)
%FIND BLINK FRAMES Records in a mat file the frames in which a blink
%occurred.
%   The result is stored with '_blinkframes' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

stimLocsMatFileName = [inputVideoPath(1:end-4) '_stimlocs'];
badFramesMatFileName = [inputVideoPath(1:end-4) '_blinkframes'];

%% Handle overwrite scenarios.
if ~exist([badFramesMatFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['FindBadFrames() did not execute because it would overwrite existing file. (' badFramesMatFileName ')'], parametersStructure);
    return;
else
    RevasWarning(['FindBadFrames() is proceeding and overwriting an existing file. (' badFramesMatFileName ')'], parametersStructure);
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'thresholdValue')
    thresholdValue = 0.8;
    RevasWarning('using default parameter for thresholdValue', parametersStructure);
else
    thresholdValue = parametersStructure.thresholdValue;
end

%% Identify bad frames
v = VideoReader(inputVideoPath);
means = zeros(1, v.FrameRate*v.Duration);
frameNumber = 1;

% First find the average pixel value of each individual frame
% We must compute this again because there is no gaurantee that a
% |stimlocs| file exists and even if it does, we cannot say for sure that
% those values are the most updated.
while hasFrame(v)
    frame = readFrame(v);
    try 
        frame = rgb2gray(frame);
    catch
    end
    mean = sum(sum(frame))/(v.Height*v.Width);
    means(1, frameNumber) = mean;
    frameNumber = frameNumber + 1;
end

% Then find the standard deviation of the average pixel values
standardDeviation = std2(means);

% Mark frames that are beyond our threshold as bad frames
threshold = thresholdValue * standardDeviation;
badFrames = zeros(1, size(means, 2));
lowerBound = mean2(means) - threshold;
upperBound = mean2(means) + threshold;

for frameNumber = 1:size(means, 2)
    kthMean = means(1, frameNumber);
    if parametersStructure.upperTail == true
        if kthMean > upperBound
            badFrames(frameNumber) = 1;
        end
    else
        if kthMean < lowerBound
            badFrames(frameNumber) = 1;
        end
    end
end

%% Lump together blinks that are < |stitchCriteria| frames apart

% If the difference between any two marked saccades is less than
% |stitchCriteria|, then lump them together as one.
if isfield(parametersStructure, 'stitchCriteria')
    badFramesIndices = find(badFrames);
    badFramesDiffs = diff(badFramesIndices);
    for i = 1:size(badFramesDiffs, 2)
        if badFramesDiffs(i) > 1 && badFramesDiffs(i) < parametersStructure.stitchCriteria
            for j = 1:badFramesDiffs(i)
                badFrames(badFramesIndices(i)+j) = 1;
            end
        end
    end
end

%% Filter out leftover 0 padding
badFrames = find(badFrames);
badFrames = badFrames(badFrames ~= 0);

%% Save to output mat file
save(badFramesMatFileName, 'badFrames');

end
