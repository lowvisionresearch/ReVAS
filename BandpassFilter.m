function BandpassFilter(inputVideoPath, parametersStructure)
%BANDPASS FILTER Applies bandpass filtering to the video
%   The result is stored with '_bpfiltered' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputVideoPath = [inputVideoPath(1:end-4) '_bpfiltered' inputVideoPath(end-3:end)];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    warning('RemoveStimuli() did not execute because it would overwrite existing file.');
    return;
else
    warning('RemoveStimuli() is proceeding and overwriting an existing file.');
end

%% Set bandpassSigma

if ~isfield(parametersStructure, 'bandpassSigma')
    bandpassSigma = 0.6;
else
    bandpassSigma = parametersStructure.bandpassSigma;
end

%% Gamma correct frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

numberOfFrames = size(videoInputArray, 3);

for frameNumber = 1:numberOfFrames
    
    videoInputArray(:,:,frameNumber) = ...
        imgaussfilt(videoInputArray(:,:,frameNumber), bandpassSigma);
    
end

writeVideo(writer, videoInputArray);
close(writer);

end

