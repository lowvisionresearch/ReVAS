function [outputVideo, varargout] = TrimVideo(inputVideo, params)
%TRIM VIDEO Removes boundaries of video. 
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is either the path to the video, or the video matrix itself. 
%   In the former situation, the result is that the
%   trimmed version of this video is written with '_trim' appended to the
%   original file name. In the latter situation, no video is written and
%   the result is returned.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
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
%  badFrames            : specifies blink/bad frames. we can skip those but
%                         we need to make sure to keep a record of 
%                         discarded frames.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is path to new video if 'inputVideo' is also a path. If
%   'inputVideo' is a 3D array, |outputVideo| is also a 3D array.
%
%   |varargout| is a variable output argument holder. Used to return the 
%   'params' structure. 
%
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       params.overwrite = 1;
%       params.borderTrimAmount = [0 0 12 0];
%       params.badFrames = false;
%       TrimVideo(inputVideo, params);

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end


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
    params = struct;
end

% validate params
[~,callerStr] = fileparts(mfilename);
[default, validate] = GetDefaults(callerStr);
params = ValidateField(params,default,validate,callerStr);


%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = Filename(inputVideo, 'trim');
    params.outputVideoPath = outputVideoPath;
    
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~params.overwrite
        RevasWarning(['TrimVideo() did not execute because it would overwrite existing file. (' outputVideoPath ')'], params);    
        outputVideo = outputVideoPath;
        
        if nargout > 1
            varargout{1} = params;
        end
        return;
    else
        RevasWarning(['TrimVideo() is proceeding and overwriting an existing file. (' outputVideoPath ')'], params);  
    end
end


%% Create reader/writer objects and get some info on videos

left = params.borderTrimAmount(1);
right = params.borderTrimAmount(2);
top = params.borderTrimAmount(3);
bottom = params.borderTrimAmount(4);

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
    
    % preallocate the output video array
    outputVideo = zeros(height-(top+bottom), width-(left+right), ...
        numberOfFrames-sum(params.badFrames),'uint8');
end


%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);


%% Write out new video or return a 3D array
    
% Read, trim, and write frame by frame.
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
        if params.skipFrame(fr)
            continue;
        end

        % trim
        frame = frame(top+1 : height-bottom, ...
            left+1 : width-right);

        % write out
        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~params.badFrames(1:fr));
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

%% return the params structure if requested
if nargout > 1
    varargout{1} = params;
end

