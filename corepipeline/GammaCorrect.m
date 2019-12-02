function outputVideo = GammaCorrect(inputVideo, parametersStructure)
%GAMMA CORRECT Applies gamma correction or some other contrast enhancement
% operation to the video.
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
%   method             : 'simpleGamma' for simple gamma correction, 
%                        'toneMapping' for boosting only low-mid grays, 
%                        'histEq' for histogram equalization. (default
%                        'simpleGamma'). Only one of the methods can be
%                        applied with a single call to this function.
%   gammaExponent      : (applies only when 'simpleGamma' method is 
%                        selected) gamma specifies the shape of the curve 
%                        describing the relationship between the 
%                        values in I and J, where new intensity
%                        values are being mapped from I (a frame) 
%                        to J. gammaExponent is a scalar value.
%                        (default 0.6)
%   toneCurve          : a 256x1 uint8 array for mapping pixel values to
%                        new values by simply indexing toneCurve using
%                        video frame itself. 
%   histLevels         : number of levels to be used in histeq. (default 64)
%   badFrames          : specifies blink/bad frames. we can skip those but
%                        we need to make sure to keep a record of 
%                        discarded frames. 
%   Example usage: 
%       inputVideo = 'tslo-dark.avi';
%       parametersStructure.overwrite = true;
%       parametersStructure.method = 'simpleGamma';
%       parametersStructure.gammaExponent = 0.6;
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

%% Set parameters to defaults if not specified.

if nargin < 2
    parametersStructure = struct;
end

if ~isfield(parametersStructure, 'overwrite')
    overwrite = false; 
else
    overwrite = parametersStructure.overwrite;
end

if ~isfield(parametersStructure, 'method')
    method = 'simpleGamma';
    RevasWarning(['GammaCorrect is using default parameter for method: ' method] , parametersStructure);
else
    method = parametersStructure.method;
end

if ~isfield(parametersStructure, 'gammaExponent')
    gammaExponent = 0.6;
    RevasWarning(['GammaCorrect is using default parameter for gammaExponent: ' num2str(gammaExponent)] , parametersStructure);
else
    gammaExponent = parametersStructure.gammaExponent;
end

if ~isfield(parametersStructure, 'toneCurve')
    toneCurve = uint8(0:255); % this does nothing 
    RevasWarning('GammaCorrect is using default parameter for toneCurve: no correction.' , parametersStructure);
else
    toneCurve = parametersStructure.toneCurve;
end

if ~isfield(parametersStructure, 'histLevels')
    histLevels = 64;
    RevasWarning(['GammaCorrect is using default parameter for histLevels: ' num2str(histLevels)], parametersStructure);
else
    histLevels = parametersStructure.histLevels;
end

if ~isfield(parametersStructure, 'badFrames')
    badFrames = false;
    RevasWarning('GammaCorrect is using default parameter for badFrames: none.', parametersStructure);
else
    badFrames = parametersStructure.badFrames;
end


%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = Filename(inputVideo, 'gamma');
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~overwrite
        RevasWarning(['GammaCorrect() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
        return;
    else
        RevasWarning(['GammaCorrect() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
    end
end



%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Create reader/writer objects and get some info on videos

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    reader = VideoReader(inputVideo);
    % some videos are not 30fps, we need to keep the same framerate as
    % the source video.
    writer.FrameRate=reader.Framerate;
    open(writer);

    % Determine dimensions of video.
    numberOfFrames = reader.Framerate * reader.Duration;
else
    % Determine dimensions of video.
    [height, width, numberOfFrames] = size(inputVideo);
    
    % preallocate the output video array
    outputVideo = zeros(height, width, numberOfFrames-sum(badFrames),'uint8');

end

%% badFrames handling
% If badFrames is not provided, use all frames
if length(badFrames)<=1 && ~badFrames
    badFrames = false(numberOfFrames,1);
end

% If badFrames are provided but its size don't match the number of frames
if length(badFrames) ~= numberOfFrames
    badFrames = false(numberOfFrames,1);
    RevasWarning('GammaCorrect(): size mismatch between ''badFrames'' and input video. Using all frames for this video.', parametersStructure);  
end


%% Contrast enhancement

if strcmpi(method,'simpleGamma')
    simpleGammaToneCurve = uint8((linspace(0,1,256).^gammaExponent)*255);
end

% Read, do contrast enhancement, and write frame by frame.
for fr = 1:numberOfFrames
    if ~abortTriggered

        % get next frame
        if writeResult
            frame = readFrame(reader);
            if ndims(frame) == 3
                frame = rgb2gray(frame);
            end
        else
            frame = inputVideo(:,:, fr);
        end

        % if it's a blink frame, skip it.
        if badFrames(fr)
            continue;
        end

        % apple contrast enhancement here
        switch method
            case 'simpleGamma'
                frame = simpleGammaToneCurve(frame+1);
                
            case 'toneMapping'
                frame = toneCurve(frame+1);
                
            case 'histEq'
                frame = uint8(histeq(frame, histLevels));
                
            otherwise
                error('unknown method type for contrast GammaCorrect().');
        end

        % write out
        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~badFrames(1:fr));
            outputVideo(:, :, nextFrameNumber) = frame; 
        end
    end
end % end of video

%% return results, close up objects

if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if abortTriggered
        delete(outputVideoPath)
    end
end


