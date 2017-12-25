function BandpassFilter(inputVideoPath, parametersStructure)
%BANDPASS FILTER Applies bandpass filtering to the video.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video. The result is stored with 
%   '_bandfilt' appended to the input video file name.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
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
%                             
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = true;
%       parametersStructure.smoothing = 1;
%       parametersStructure.lowSpatialFrequencyCutoff = 3;
%       BandpassFilter(inputVideoPath, parametersStructure);

%% Handle overwrite scenarios.
outputVideoPath = [inputVideoPath(1:end-4) '_bandfilt' inputVideoPath(end-3:end)];
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['BandpassFilter() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
    return;
else
    RevasWarning(['BandpassFilter() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);  
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'smoothing')
    smoothing = 1; % standard deviation of the gaussian kernel, in pixels
    RevasWarning('using default parameter for smoothing', parametersStructure);
else
    smoothing = parametersStructure.smoothing;
    if ~IsNaturalNumber(smoothing)
        error('smoothing must be a natural number');
    end
end

if ~isfield(parametersStructure, 'lowSpatialFrequencyCutoff')
    lowSpatialFrequencyCutoff = 3; % cycles per image
    RevasWarning('using default parameter for lowSpatialFrequencyCutoff', parametersStructure);
else
    lowSpatialFrequencyCutoff = parametersStructure.lowSpatialFrequencyCutoff;
    if ~IsNonNegativeRealNumber(lowSpatialFrequencyCutoff)
        error('lowSpatialFrequencyCutoff must be a non-negative real number');
    end
end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Bandpass filter frame by frame

% create a video writer object and open it.
writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

% Determine dimensions of video.
reader = VideoReader(inputVideoPath);
width = reader.Width;
height = reader.Height;
numberOfFrames = reader.Framerate * reader.Duration;

% create pixel position arrays. 
xVector = (0:width - 1) - floor(width / 2); 
yVector = flipud((0:height - 1)') - floor(height / 2); 
radiusMatrix = sqrt((repmat(xVector,height,1) .^ 2) + ...
                    (repmat(yVector,1,width) .^ 2));

% create the amplitude response of the high-pass filter (which will remove
% only the low spatial frequency components in the image such as luminance
% gradient, darker foveal pit, etc.)
highPassFilter = double(radiusMatrix > lowSpatialFrequencyCutoff);
highPassFilter(floor(height/2) + 1, floor(width/2) + 1) = 1;

% Read, apply filters, and write frame by frame.
for frameNumber = 1:numberOfFrames
    if ~abortTriggered
        frame = readFrame(reader);
        if ndims(frame) == 3
            frame = rgb2gray(frame);
        end
    
        % apply smoothing
        I1 = imgaussfilt(frame, smoothing);

        % remove low spatial frequencies
        I2 = abs(ifft2((fft2(I1)) .* ifftshift(highPassFilter)));

        % normalize to maximize contrast
        maxMin = [max(I2(:))  min(I2(:))];
        rangeOfValues = abs(diff(maxMin));
        frame = uint8(255*(I2 - maxMin(2))/rangeOfValues);

        writeVideo(writer, frame);
    end
end

close(writer);

end
