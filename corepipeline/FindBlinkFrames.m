function FindBlinkFrames(inputVideoPath, parametersStructure)
%FIND BLINK FRAMES  Records in a mat file the frames in which a blink
%                   occurred. Blinks are considered to be frames in which
%                   the frame's mean pixel value is above or below a 
%                   specified number of standard deviations from the mean
%                   of the mean pixel values of all frames. 
%   The result is stored with '_blinkframes' appended to the input video file
%   name.
%
%
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  overwrite           :   set to 1 to overwrite existing files resulting 
%                          from calling FindBlinkFrames.
%                          Set to 0 to abort the function call if the
%                          files exist in the current directory.
%  thresholdValue      :   the number of standard deviations from the mean
%                          of the mean frame pixel values above or below
%                          which a frame is designated a blink frame. This
%                          can be a decimal number. Defaults to 1.0 if no 
%                          value is specified.
%  upperTail           :   set to true if blinks in the video show up as
%                          brighter than non-blink frames. Set to false if 
%                          blinks are black frames or darker than non-blink
%                          frames (this is usually the case). This
%                          parameter determines whether to flag bad frames
%                          as those above or below the thresholdValue
%                          (upperTail set to true flags frames above the
%                          thresholdValue, upperTail set to false flags
%                          frames below the thresoldValue).
%  stitchCriteria      :   optional--specify the maximum distance (in frames)
%                          between blinks, below which two blinks will be
%                          marked as one. For example, if badFrames is 
%                          [8, 9, 11, 12], this represents two blinks, one
%                          at frames 8 and 9, and the other at frames 
%                          11 and 12. If stitchCriteria is 2, then 
%                          badFrames becomes [8, 9, 10, 11, 12] because the
%                          distance between the blinks [8, 9] and [11, 12]
%                          are separated by only one frame, which is less
%                          than the specified stitch criteria.
%                          
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       videoPath = 'MyVid.avi';
%       parametersStructure.overwrite = true;
%       parametersStructure.threhsoldValue = 1.5;
%       parametersStructure.upperTail = false;
%       FindBlinkFrames(videoPath, parametersStructure);

%% Handle overwrite scenarios.
stimLocsMatFileName = [inputVideoPath(1:end-4) '_stimlocs'];
badFramesMatFileName = [inputVideoPath(1:end-4) '_blinkframes'];
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
    thresholdValue = 1.0;
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
