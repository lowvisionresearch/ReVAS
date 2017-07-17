function BandpassFilter(inputVideoPath, parametersStructure)
%BANDPASS FILTER Applies bandpass filtering to the video
%   The result is stored with '_bandfilt' appended to the input video file
%   name.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite                   Determines whether an existing output file
%                               should be overwritten and replaced if it
%                               already exists.(default false)
%   smoothing                   Used to remove high-frequency noise in the
%                               frames. Represents the standard deviation
%                               of a Gaussian kernel, in pixels.(default 1)
%   lowSpatialFrequencyCutoff   Used to remove low-frequency fluctuations
%                               in the frames which messes up strip
%                               analysis. For instance, brightness
%                               gradients due to the way observer's head is
%                               positioned in the TSLO, or just the darker
%                               nature of the foveal pit compared to the
%                               peripheral retina creates these low-freq.
%                               fluctuations.(default 3 cycles/image)

outputVideoPath = [inputVideoPath(1:end-4) '_bandfilt' inputVideoPath(end-3:end)];

%% Handle overwrite scenarios.
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['BandpassFilter() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
    return;
else
    RevasWarning(['BandpassFilter() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);  
end

%% Set smoothing and lowSpatialFrequencyCutoff

if ~isfield(parametersStructure, 'smoothing')
    smoothing = 1; % standard deviation of the gaussian kernel, in pixels
else
    smoothing = parametersStructure.smoothing;
end

if ~isfield(parametersStructure, 'lowSpatialFrequencyCutoff')
    lowSpatialFrequencyCutoff = 3; % cycles per image
else
    lowSpatialFrequencyCutoff = parametersStructure.lowSpatialFrequencyCutoff;
end

if smoothing < 0
    error('smoothing should not be non-negative');
end

if lowSpatialFrequencyCutoff < 0
    error('lowSpatialFrequencyCutoff should not be non-negative');
end

%% Bandpass filter frame by frame

% create a video writer object and open it.
writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

% read all frames and store them in memory (very inefficient for big videos)
[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

% get the number of frames and their sizes
[height, width, numberOfFrames] = size(videoInputArray);

% create pixel position arrays. 
xVector = (0:width - 1) - floor(width / 2); 
yVector = flipud((0:height - 1)') - floor(height / 2); 
radiusMatrix = sqrt((repmat(xVector,height,1) .^ 2) + ...
                    (repmat(yVector,1,width) .^ 2));

% create the amplitude response of the high-pass filter (which will remove
% only the low spatial frequency components in the image such as luminance
% gradient, darker foveal pit, etc.)
highPassFilter = double(radiusMatrix > lowSpatialFrequencyCutoff);
highPassFilter(floor(height / 2)+1,floor(width / 2)+1) = 1;

% now it's time to apply filters to each and every frame
for frameNumber = 1:numberOfFrames
    
    % get a single frame
    I = videoInputArray(:,:,frameNumber);
    
    % apply smoothing
    I1 = imgaussfilt(I, smoothing);
    
    % remove low spatial frequencies
    I2 = abs(ifft2( (fft2(I1)) .* ifftshift(highPassFilter) ));
    
    % normalize to maximize contrast
    maxMin = [max(I2(:))  min(I2(:))];
    rangeOfValues = abs(diff(maxMin));
    newFrame = uint8(255*(I2 - maxMin(2))/rangeOfValues);
    
    % update the video array 
    videoInputArray(:,:,frameNumber) = newFrame;
    
end

writeVideo(writer, videoInputArray);
close(writer);

end

