function GammaCorrect(inputVideoPath, parametersStructure)
%GAMMA CORRECT Applies gamma correction to the video.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video. The result is stored with 
%   '_gamscaled' appended to the input video file name.
%
%   |parametersStructure| is a struct as specified below.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite          : set to true to overwrite existing files.
%                        Set to false to abort the function call if the
%                        files already exist. (default false)
%   gammaExponent      : gamma specifies the shape of the curve 
%                        describing the relationship between the 
%                        values in I and J, where new intensity
%                        values are being mapped from I (a frame) 
%                        to J. gammaExponent is a scalar value.
%                        (default 0.6)
%
%   Example usage: 
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = true;
%       parametersStructure.gammaExponent = 0.6;
%       GammaCorrect(inputVideoPath, parametersStructure);

%% Handle overwrite scenarios.
outputVideoPath = [inputVideoPath(1:end-4) '_gamscaled' inputVideoPath(end-3:end)];
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['GammaCorrect() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
    return;
else
    RevasWarning(['GammaCorrect() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'gammaExponent')
    gammaExponent = 0.6;
    RevasWarning('using default parameter for gammaExponent', parametersStructure);
else
    gammaExponent = parametersStructure.gammaExponent;
    if ~IsRealNumber(gammaExponent)
       error('gammaExponent must be a real number'); 
    end
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Gamma correct frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
reader = VideoReader(inputVideoPath);
% some videos are not 30fps, we need to keep the same framerate as
% the source video.
writer.FrameRate=reader.Framerate;
open(writer);

% Determine dimensions of video.
numberOfFrames = reader.Framerate * reader.Duration;

% Read, gamma correct, and write frame by frame.
for frameNumber = 1:numberOfFrames
    if ~abortTriggered
        frame = readFrame(reader);
        if ndims(frame) == 3
            frame = rgb2gray(frame);
        end
        frame = imadjust(frame, [], [], gammaExponent);
       writeVideo(writer, frame);
    end
end

close(writer);

end

