function [outputVideo, params] = TrimVideo(inputVideo, params)
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
%                         Set to false to params.abort the function call if the
%                         files already exist. (default false)
%   enableVerbosity     : true/false. if true, plots first frame after filtering.
%   borderTrimAmount    : specifies the number of rows and columns to be
%                         removed as a vector with the number of
%                         rows/columns to be removed from each edge
%                         specified in the following order:
%                         [left right top bottom]. The default is
%                         removing 24 from the right and top. If a scalar
%                         is provided instead, then that amount will be
%                         removed from the right and top only.
%                         (default [0 24 24 0])
%   badFrames           : specifies blink/bad frames. we can skip those but
%                         we need to make sure to keep a record of 
<<<<<<< HEAD
%                         discarded frames. (default empty)
%   axesHandles         : handles to an axes object. (default empty)
=======
%                         discarded frames.
%   axesHandles         : handles to an axes object.
>>>>>>> db15085144bcbcbb17a80d2adad70381819db415
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is path to new video if 'inputVideo' is also a path. If
%   'inputVideo' is a 3D array, |outputVideo| is also a 3D array.
%
%   |params| structure.
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


%% Handle GUI mode
% params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
    isGUI = true;
else
    logBox = [];
    isGUI = false;
end

% params will have access to a uicontrol object in GUI mode. so if it does
% not already have that, create the field and set it to false so that this
% module can be used without the GUI
if ~isfield(params,'abort')
    params.abort.Value = false;
end

%% Handle verbosity 
if ischar(params.enableVerbosity)
    params.enableVerbosity = find(contains({'none','video','frame'},params.enableVerbosity))-1;
end

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Trimming',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.4 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');
    params.axesHandles(1) = subplot(1,1,1);
end

if params.enableVerbosity
    cla(params.axesHandles(1))
    tb = get(params.axesHandles(1),'toolbar');
    tb.Visible = 'on';
end


%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = Filename(inputVideo, 'trim');
    params.outputVideoPath = outputVideoPath;
    
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~params.overwrite
        
        RevasWarning(['TrimVideo() did not execute because it would overwrite existing file. (' outputVideoPath ')'], logBox);    
        outputVideo = outputVideoPath;
        return;
    else
        RevasWarning(['TrimVideo() is proceeding and overwriting an existing file. (' outputVideoPath ')'], logBox);  
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
    if ~params.abort.Value

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

        % visualize
        if (params.enableVerbosity == 1 && fr == 1) || params.enableVerbosity > 1
            axes(params.axesHandles(1)); %#ok<LAXES>
            if fr == 1
                imh = imshow(frame,'border','tight');
            else
                imh.CData = frame;
            end
            title(params.axesHandles(1),sprintf('Trimming frames. %d out of %d',fr, numberOfFrames));
        end

        % write out
        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~params.badFrames(1:fr));
            outputVideo(:, :, nextFrameNumber) = frame; 
        end
    else 
        break;
    end
    
    if isGUI
        pause(.02);
    end
end % end of video

params.trim = params.borderTrimAmount(3:4);

%% return results, close up objects

if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if params.abort.Value
        delete(outputVideoPath)
    end
end

% remove unnecessary fields
params = RemoveFields(params,{'logBox','axesHandles','abort'}); 
