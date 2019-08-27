function outputVideo = TrimVideo(inputVideo, parametersStructure)
%TRIM VIDEO Removes upper and right edge of video.
%   Removes the upper few rows and right few columns. 
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is either the path to the video, or the video matrix itself. 
%   In the former situation, the result is that the
%   trimmed version of this video is written with '_dwt' appended to the
%   original file name. In the latter situation, no video is written and
%   the result is returned.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite           : set to true to overwrite existing files.
%                         Set to false to abort the function call if the
%                         files already exist. (default false)
%   borderTrimAmount    : specifies the number of rows and columns to be
%                         removed as a vector with the number of
%                         rows/columns to be removed from each edge
%                         specified in the following order:
%                         [left right top bottom]. The default is
%                         removing 24 from the right and top. If a scalar
%                         is provided instead, then that amount will be
%                         removed from the right and top only.
%                         (default [0 24 24 0])
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       parametersStructure.overwrite = 1;
%       parametersStructure.borderTrimAmount = [0 24 24 0];
%       TrimVideo(inputVideo, parametersStructure);

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
    outputVideoPath = Filename(inputVideo, 'trim');
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif nargin == 1 || ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasWarning(['TrimVideo() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);    
        return;
    else
        RevasWarning(['TrimVideo() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);  
    end
end

%% Set parameters to defaults if not specified.

if nargin == 1 || ~isfield(parametersStructure, 'borderTrimAmount')
    parametersStructure.borderTrimAmount = [0 24 24 0];
    RevasWarning('using default parameter for borderTrimAmount', parametersStructure);
elseif isscalar(parametersStructure.borderTrimAmount)
    parametersStructure.borderTrimAmount = [0 parametersStructure.borderTrimAmount ...
        parametersStructure.borderTrimAmount 0];
else
    parametersStructure.borderTrimAmount = parametersStructure.borderTrimAmount;
end
for t = parametersStructure.borderTrimAmount
    if ~IsNaturalNumber(t)
        error('borderTrimAmount must consist of natural numbers');
    end
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Trim the video frame by frame

left = parametersStructure.borderTrimAmount(1);
right = parametersStructure.borderTrimAmount(2);
top = parametersStructure.borderTrimAmount(3);
bottom = parametersStructure.borderTrimAmount(4);

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    reader = VideoReader(inputVideo);
    % some videos are not 30fps, we need to keep the same framerate as
    % the source video.
    writer.FrameRate = reader.Framerate;
    open(writer);
    
    % Determine dimensions of video.
    width = reader.Width;
    height = reader.Height;
    numberOfFrames = reader.Framerate * reader.Duration;
    
else
    % Determine dimensions of video.
    [height, width, numberOfFrames] = size(inputVideo);
end

if writeResult
    % Read, trim, and write frame by frame.
    for frameNumber = 1:numberOfFrames
        if ~abortTriggered

                frame = readFrame(reader);

                if ndims(frame) == 3
                    frame = rgb2gray(frame);
                end

                frame = frame(top+1 : height-bottom, ...
                    left+1 : width-right);

           writeVideo(writer, frame);
        end
    end
    
    close(writer);
else
    outputVideo = inputVideo(top+1 : height-bottom, ...
                    left+1 : width-right, ...
                    1 : end);
end
end
