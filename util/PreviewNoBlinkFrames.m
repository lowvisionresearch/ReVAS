function PreviewNoBlinkFrames()
%PREVIEW NO BLINK FRAMES Generates a video with blink frames ommitted
%   Generates a video with blink frames ommitted. This can be helpful in
%   determining whether an adequate number of blink frames have been
%   removed or whether pre-processing should be adjusted.
%   When prompted, user should select the pre-processed videos.

addpath(genpath('..'));

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

for i = 1:length(filenames)
    originalVideoPath = filenames{i};
    blinkFramesPath = [originalVideoPath(1:end-4) '_blinkframes'];
    outputVideoPath = [originalVideoPath(1:end-4) '_noblink' originalVideoPath(end-3:end)];

    load(blinkFramesPath);
    % badFrames should now be loaded.
    
    % Trim the video frame by frame.
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    [videoInputArray, ~] = VideoPathToArray(originalVideoPath);

    height = size(videoInputArray, 1);
    width = size(videoInputArray, 2);
    numberOfFrames = size(videoInputArray, 3);

    % Preallocate.
    result = zeros(height, width, ...
        numberOfFrames-size(badFrames, 2), 'uint8');
    
    resultIndex = 1;
    for frameNumber = 1:numberOfFrames
        if ~ismember(frameNumber, badFrames)
            result(:,:,resultIndex) = videoInputArray(:,:,frameNumber);
            resultIndex = resultIndex + 1;
        end
    end

    writeVideo(writer, result);
    close(writer);
    fprintf('%d of %d completed.\n', i, length(filenames));
end
fprintf('Process Completed.\n');
end
