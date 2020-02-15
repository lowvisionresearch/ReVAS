function [outputVideo, params] = BandpassFilter(inputVideo, params)
%BANDPASS FILTER Applies bandpass filtering to the video.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is either the path to the video, or the video matrix itself. 
%   In the former situation, the result is written with '_bandfilt' appended to the
%   input file name. In the latter situation, no video is written and
%   the result is returned.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite                 : set to true to overwrite existing files.
%                               Set to false to abort the function call if the
%                               files already exist. (default false)
%   smoothing                 : Used to remove high-frequency noise in the
%                               frames. Represents the standard deviation
%                               of a Gaussian kernel, in pixels. (default 1)
%   lowSpatialFrequencyCutoff : Used to remove low-frequency fluctuations
%                               in the frames which messes up strip
%                               analysis in cycles/image. For instance, brightness
%                               gradients due to the way observer's head is
%                               positioned in the TSLO, or just the darker
%                               nature of the foveal pit compared to the
%                               peripheral retina creates these low-freq.
%                               fluctuations. (default 3)
%   badFrames                 : specifies blink/bad frames. we can skip those but
%                               we need to make sure to keep a record of 
%                               discarded frames. 
%                             
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is path to new video if 'inputVideo' is also a path. If
%   'inputVideo' is a 3D array, |outputVideo| is also a 3D array.
%
%   |params| structure.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'tslo-dark.avi';
%       params.overwrite = true;
%       params.smoothing = 1;
%       params.lowSpatialFrequencyCutoff = 3;
%       BandpassFilter(inputVideo, params);

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% in GUI mode, params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
else
    logBox = [];
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
    outputVideoPath = Filename(inputVideo, 'bandpass');
    params.outputVideoPath = outputVideoPath;
    
    if ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~params.overwrite
        RevasWarning(['BandpassFilter() did not execute because it would overwrite existing file. (' outputVideoPath ')'], logBox);
        return;
    else
        RevasWarning(['BandpassFilter() is proceeding and overwriting an existing file. (' outputVideoPath ')'], logBox);  
    end
end


%% Create reader/writer objects and get some info on videos

if writeResult
    % create a video writer object and open it.
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
    outputVideo = zeros(height, width, numberOfFrames-sum(params.badFrames),'uint8');

end

%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);


%% Bandpass filter

% create pixel position arrays. 
xVector = (0:width - 1) - floor(width / 2); 
yVector = flipud((0:height - 1)') - floor(height / 2); 
radiusMatrix = sqrt((repmat(xVector,height,1) .^ 2) + ...
                    (repmat(yVector,1,width) .^ 2));

% create the amplitude response of the high-pass filter (which will remove
% only the low spatial frequency components in the image such as luminance
% gradient, darker foveal pit, etc.)
highPassFilter = double(radiusMatrix > params.lowSpatialFrequencyCutoff);
highPassFilter(floor(height/2) + 1, floor(width/2) + 1) = 1;

% Read, apply filters, and write frame by frame.
for fr = 1:numberOfFrames
    if ~abortTriggered
        
        % get next frame
        if writeResult
            frame = readFrame(reader);
            if ndims(frame) == 3
                frame = rgb2gray(frame);
            end
        else
            frame = inputVideo(1:end, 1:end, fr);
        end
        
        % if it's a blink frame, skip it.
        if params.skipFrame(fr)
            continue;
        end

        % apply params.smoothing
        I1 = imgaussfilt(frame, params.smoothing);

        % remove low spatial frequencies
        I2 = abs(ifft2((fft2(I1)) .* ifftshift(highPassFilter)));

        % normalize to maximize contrast
        maxMin = [max(I2(:))  min(I2(:))];
        rangeOfValues = abs(diff(maxMin));
        frame = uint8(255*(I2 - maxMin(2))/rangeOfValues);

        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~params.badFrames(1:fr));
            outputVideo(:, :, nextFrameNumber) = frame;
        end
    end
end

%% return results, close up objects

if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if abortTriggered
        delete(outputVideoPath)
    end
end


