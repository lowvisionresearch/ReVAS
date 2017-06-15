function BandpassFilter(inputVideoPath, parametersStructure)
%BANDPASS FILTER Applies bandpass filtering to the video
%   The result is stored with '_bandfilt' appended to the input video file
%   name.
%
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputVideoPath = [inputVideoPath(1:end-4) '_bandfilt' inputVideoPath(end-3:end)];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    warning('BandpassFilter() did not execute because it would overwrite existing file.');
    return;
else
    warning('BandpassFilter() is proceeding and overwriting an existing file.');
end

%% Set bandpassSigmaUpper and bandpassSigmaLower

if ~isfield(parametersStructure, 'bandpassSigmaUpper')
    bandpassSigmaUpper = 3;
else
    bandpassSigmaUpper = parametersStructure.bandpassSigmaUpper;
end

if ~isfield(parametersStructure, 'bandpassSigmaLower')
    bandpassSigmaLower = 25;
else
    bandpassSigmaLower = parametersStructure.bandpassSigmaLower;
end

if parametersStructure.bandpassSigmaUpper > bandpassSigmaLower
    error('bandpassSigmaUpper should not be > bandpassSigmaLower');
end

%% Gamma correct frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

numberOfFrames = size(videoInputArray, 3);

for frameNumber = 1:numberOfFrames
    
    I = videoInputArray(:,:,frameNumber);
    I1 = imgaussfilt(I, bandpassSigmaUpper);
    I2 = imgaussfilt(I1, bandpassSigmaLower);
    
    videoInputArray(:,:,frameNumber) = histeq(I1 - I2);
    
    %Solution should look like: mna_os_10_12_1_45_0_stabfix_17_36_21_409_dwt_nostim_nostim_gamscaled_bandfilt_meanrem
end

writeVideo(writer, videoInputArray);
close(writer);

end

