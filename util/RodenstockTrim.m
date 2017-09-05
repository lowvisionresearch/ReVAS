function RodenstockTrim(originalVideoPath)
%RODENSTOCK TRIM VIDEO Removes edges of video for Rodenstock videos.
%   Removes the specified number of pixels from each side of the video.
%

top = 70;
right = 10;
bottom = 34;
left = 46;

addpath(genpath('..'));

outputVideoPath = [originalVideoPath(1:end-4) '_dwt' originalVideoPath(end-3:end)];

% Trim the video frame by frame.
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
