function [outputVideo, params, varargout] = DummyModule(inputVideo, params)
%[outputVideo, params, varargout] = DummyModule(inputVideo, params)
%
%   This is a dummy module, used to demonstrate how to add a new module to
%   ReVAS. Please see the wiki page at
%   https://github.com/lowvisionresearch/ReVAS/wiki/Module-Guidelines. It
%   takes a video (or a path to it) as first input, inverts pixel values of
%   the video and returns the modified video (or its new path) as output.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is the path to the video or a matrix representation of the
%   video that is already loaded into memory.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%
%   overwrite         : set to true to overwrite existing files. Set to 
%                       false to params.abort the function call if the files
%                       already exist. (default false)
%   enableVerbosity   : set to true to report back plots during execution.(
%                       default false)
%   badFrames         : vector containing the frame numbers of the blink 
%                       frames. (default [])
%   axesHandles       : axes handle for giving feedback. if not provided, 
%                       new figures are created. (relevant only when
%                       enableVerbosity is true)
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is the processed video (or full path to it).
%
%   |params| structure, including a new field called `randomFrame`.
%
%   |varargout| is a variable output argument holder.
%   varargout{1} = randomFrameNumber, the frame number of the randomly 
%   selected video array
% 

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
    set(fh,'name','Dummy',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.4 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');
    params.axesHandles(1) = subplot(1,1,1);
end

% clear axes
if params.enableVerbosity
    cla(params.axesHandles(1))
    tb = get(params.axesHandles(1),'toolbar');
    tb.Visible = 'on';
end


%% Handle overwrite scenarios.
if writeResult
    outputVideoPath = Filename(inputVideo, 'dummy');
    matFilePath = Filename(inputVideo, 'randomframe');
    params.outputVideoPath = outputVideoPath;
    params.matFilePath = matFilePath;
    
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        
        RevasMessage(['DummyModule() did not execute because it would overwrite existing file. (' outputVideoPath ')'], logBox);
        RevasMessage('DummyModule() is returning results from existing file.',logBox); 
        
        % try loading existing file contents
        load(matFilePath,'randomFrame','randomFrameNumber');
        params.randomFrame = randomFrame;
        if nargout > 2
            varargout{1} = randomFrameNumber;
        end
        
        return;
    else
        RevasMessage(['DummyModule() is proceeding and overwriting an existing file. (' outputVideoPath ')'], logBox);  
    end
else
    outputVideoPath = [];
    matFilePath = [];
end



%% Create a reader object if needed and get some info on video

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    % Determine dimensions of video.
    reader = VideoReader(inputVideo);
    params.frameRate = reader.FrameRate;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    [height, width, numberOfFrames] = size(inputVideo); 
    
    % preallocate the output video array
    outputVideo = zeros(height, width, numberOfFrames-sum(params.badFrames),'uint8');
end


%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);

%% select a frame randomly
randomFrameNumber = randi(numberOfFrames,1);

%% Write out new video or return a 3D array

% Read, invert, and write frame by frame.
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

        % invert pixel values
        frame = 255 - frame;
        
        % get random frame if it is the right frame number
        if randomFrameNumber == fr
            randomFrame = frame;
        end

        % visualize
        if (params.enableVerbosity == 1 && fr == 1) || params.enableVerbosity > 1
            axes(params.axesHandles(1)); %#ok<LAXES>
            if fr == 1
                imh = imshow(frame,'border','tight');
            else
                imh.CData = frame;
            end
            title(params.axesHandles(1),sprintf('Inverting frames. %d out of %d',fr, numberOfFrames));
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


%% return results, close up objects

% video -- primary output
if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if params.abort.Value
        delete(outputVideoPath)
    end
end

% additional outputs
params.randomFrame = randomFrame;

if nargout > 2 
    varargout{1} = randomFrameNumber;
end

% remove unnecessary (GUI related) fields
abort = params.abort.Value;
params = RemoveFields(params,{'logBox','axesHandles','abort'}); 

% save additional outputs
if writeResult && ~abort
    save(matFilePath,'randomFrame','randomFrameNumber','params');
end

