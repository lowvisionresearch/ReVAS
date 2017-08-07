function RodenstockTrim()
%TRIM VIDEO Removes upper and right edge of video
%   Removes the specified number of pixels from each side of the video.
%
%   |inputVideoPath| is the path to the video. The result is that the
%   trimmed version of this video is stored with '_dwt' appended to the
%   original file name.

top = 70;
right = 10;
bottom = 34;
left = 36;

addpath(genpath('..'));

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

parfor i = 1:length(filenames)
    originalVideoPath = filenames{i};
    outputVideoPath = [originalVideoPath(1:end-4) '_dwt' originalVideoPath(end-3:end)];

    % Trim the video frame by frame

    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    [videoInputArray, ~] = VideoPathToArray(originalVideoPath);

    height = size(videoInputArray, 1);
    width = size(videoInputArray, 2);
    numberOfFrames = size(videoInputArray, 3);

    % Preallocate.
    trimmedFrames = zeros(height - top - bottom, ...
        width - left - right, numberOfFrames, 'uint8');

    for frameNumber = 1:numberOfFrames
        frame = videoInputArray(:,:,frameNumber);
        trimmedFrames(:,:,frameNumber) = ...
            frame(top+1 : height-bottom, ...
           left+1 : width-right);
    end

    writeVideo(writer, trimmedFrames);
    close(writer);
end
fprintf('Process Completed.\n');
end
