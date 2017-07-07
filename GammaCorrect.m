function GammaCorrect(inputVideoPath, parametersStructure)
%GAMMA CORRECT Applies gamma correction to the video
%   The result is stored with '_gamscaled' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputVideoPath = [inputVideoPath(1:end-4) '_gamscaled' inputVideoPath(end-3:end)];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning('GammaCorrect() did not execute because it would overwrite existing file.');
    return;
else
    RevasWarning('GammaCorrect() is proceeding and overwriting an existing file.');
end

%% Set gammaExponent

if ~isfield(parametersStructure, 'gammaExponent')
    gammaExponent = 0.6;
else
    gammaExponent = parametersStructure.gammaExponent;
end

%% Gamma correct frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

numberOfFrames = size(videoInputArray, 3);

for frameNumber = 1:numberOfFrames
    
    videoInputArray(:,:,frameNumber) = ...
        imadjust(videoInputArray(:,:,frameNumber), [], [], gammaExponent);
    
end

writeVideo(writer, videoInputArray);
close(writer);

end

