function outputVideo = GammaCorrect(inputVideo, parametersStructure)
%GAMMA CORRECT Applies gamma correction to the video.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is either the path to the video, or the video matrix itself. 
%   In the former situation, the result is written with '_gamscaled' appended to the
%   input file name. In the latter situation, no video is written and
%   the result is returned.
%
%   |parametersStructure| is a struct as specified below.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite          : set to true to overwrite existing files.
%                        Set to false to abort the function call if the
%                        files already exist. (default false)
%   isHistEq           : true/false. if true, we do histogram equalization
%                        using MATLAB's histeq function. This is more
%                        robust against fluctuations in overall brightness
%                        across frames.
%   isGammaCorrect     : true/false. Note that if both methods are set to
%                        true, then gamma correction is applied first.
%   gammaExponent      : (applies only when 'isGammaCorrect' is true) 
%                        gamma specifies the shape of the curve 
%                        describing the relationship between the 
%                        values in I and J, where new intensity
%                        values are being mapped from I (a frame) 
%                        to J. gammaExponent is a scalar value.
%                        (default 0.6)
%
%   Example usage: 
%       inputVideo = 'MyVid.avi';
%       parametersStructure.overwrite = true;
%       parametersStructure.gammaExponent = 0.6;
%       parametersStructure.isGammaCorrect = true;
%       parametersStructure.isHistEq = false;
%       GammaCorrect(inputVideo, parametersStructure);

%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
end

%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = [inputVideo(1:end-4) '_gamscaled' inputVideo(end-3:end)];
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasWarning(['GammaCorrect() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
        return;
    else
        RevasWarning(['GammaCorrect() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
    end
end

%% Set parameters to defaults if not specified.
if ~isfield(parametersStructure, 'isHistEq')
    isHistEq = true;
    RevasWarning('using default parameter for isHistEq', parametersStructure);
else
    isHistEq = parametersStructure.isHistEq;
end

if ~isfield(parametersStructure, 'isGammaCorrect')
    isGammaCorrect = false;
    RevasWarning('using default parameter for isGammaCorrect', parametersStructure);
else
    isGammaCorrect = parametersStructure.isGammaCorrect;
end

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

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    reader = VideoReader(inputVideo);
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

            if isGammaCorrect
                frame = imadjust(frame, [], [], gammaExponent);
            end
            if isHistEq
                frame = histeq(frame);
            end

            writeVideo(writer, frame);
        end
    end
    
    close(writer);

else
    outputVideo = inputVideo;
    for i = 1:size(inputVideo, 3)
        if isGammaCorrect
            outputVideo(1:end,1:end,i) = imadjust(outputVideo(1:end,1:end,i), [], [], gammaExponent);
        end
        if isHistEq
            outputVideo(1:end,1:end,i) = histeq(outputVideo(1:end,1:end,i));
        end
    end
end

end
