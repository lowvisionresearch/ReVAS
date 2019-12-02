function outputVideo = TrimVideo(inputVideo, params)
%TRIM VIDEO Removes boundaries of video. 
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
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       params.overwrite = 1;
%       params.borderTrimAmount = [0 0 12 0];
%       params.badFrames = false;
%       TrimVideo(inputVideo, params);

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

if ~isfield(params, 'overwrite')
    overwrite = false; 
else
    overwrite = params.overwrite;
end

if ~isfield(params, 'borderTrimAmount')
    borderTrimAmount = [0 0 12 0];
    RevasWarning(['TrimVideo is using default parameter for borderTrimAmount: ' num2str(borderTrimAmount)], params);
else
    borderTrimAmount = params.borderTrimAmount;
    if isscalar(borderTrimAmount)
        params.borderTrimAmount = [0 borderTrimAmount borderTrimAmount 0];
        borderTrimAmount = params.borderTrimAmount;
    end
    
    % light error checking
    if any(~IsNaturalNumber(borderTrimAmount))
        error('borderTrimAmount must consist of natural numbers');
    end
end

if ~isfield(params, 'badFrames')
    badFrames = false;
    RevasWarning('TrimVideo is using default parameter for badFrames: none.', params);
else
    badFrames = params.badFrames;
end



%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = Filename(inputVideo, 'trim');
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~overwrite
        RevasWarning(['TrimVideo() did not execute because it would overwrite existing file. (' outputVideoPath ')'], params);    
        return;
    else
        RevasWarning(['TrimVideo() is proceeding and overwriting an existing file. (' outputVideoPath ')'], params);  
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

left = borderTrimAmount(1);
right = borderTrimAmount(2);
top = borderTrimAmount(3);
bottom = borderTrimAmount(4);

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
        numberOfFrames-sum(badFrames),'uint8');
end


%% badFrames handling
% If badFrames is not provided, use all frames
if length(badFrames)<=1 && ~badFrames
    badFrames = false(numberOfFrames,1);
end

% If badFrames are provided but its size don't match the number of frames
if length(badFrames) ~= numberOfFrames
    badFrames = false(numberOfFrames,1);
    RevasWarning(['TrimVideo(): size mismatch between ''badFrames'' and input video. Using all frames for (' outputVideoPath ')'], params);  
end


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
        if badFrames(fr)
            continue;
        end

        % trim
        frame = frame(top+1 : height-bottom, ...
            left+1 : width-right);

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


