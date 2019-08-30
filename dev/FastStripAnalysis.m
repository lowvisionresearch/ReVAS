function [xPos, yPos] = FastStripAnalysis(pathToVideo, isGPU, isVisualize)

if nargin < 1 || isempty(pathToVideo)
    pathToVideo = 'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi';
end

if nargin < 2 || isempty(isGPU)
    isGPU = false;
end

if nargin < 3 || isempty(isVisualize)
    isVisualize = 1;
end

method = 'fft';
downSampleFactor = 2;

% create a video reader object
videoObj = VideoReader(pathToVideo);
startTime = videoObj.CurrentTime;

% use the first frame as the reference frame
%refFrame = imresize(WhereToCompute(single(readFrame(videoObj))/255, isGPU),...
%    1/downSampleFactor);

% load a reference frame
load('demo/sample10deg_dwt_nostim_gamscaled_bandfilt_refframe.mat', 'refFrame');
refFrame = imresize(WhereToCompute(single(refFrame)/255, isGPU), 1/downSampleFactor);

% rewind back to the beginning of the video
videoObj.CurrentTime = startTime;

% define strip parameters
stripHeight = ceil(15/downSampleFactor);
stripWidth = ceil(videoObj.Width/downSampleFactor);
numberOfStrips = 18;
delta = 1;
stripLocations = round(linspace(delta, ...
    videoObj.Height/downSampleFactor - stripHeight + 1, numberOfStrips));

% precomputed arrays
mask = WhereToCompute(ones(stripHeight, stripWidth,'single'), isGPU);
fuv = conv2(refFrame,mask);
f2uv = conv2(refFrame.^2,mask);

% precision of the computations
eps = 10^-6;

% energy of the reference
euv = (f2uv - (fuv.^2)/(stripHeight * stripWidth));
euv(euv == 0) = eps;

% shift, sqrt, and take the reciprocal of euv here for speed up. Note that
% division is more expensive than multiplication.
ieuv = 1./sqrt(circshift(euv, -[stripHeight stripWidth]+1));

[refFrameHeight, refFrameWidth] = size(refFrame);

% fft of the reference
cm = stripHeight + refFrameHeight - 1;
cn = stripWidth  + refFrameWidth  - 1;
fr = fft2(refFrame, cm, cn);

% preallocate arrays
xPos = WhereToCompute(nan(videoObj.FrameRate * videoObj.Duration * numberOfStrips,1),...
    isGPU);
yPos = xPos;
sampleCounter = 0;

% analyze video
t0 = tic;
while hasFrame(videoObj)
    
    currentFrame = imresize( ...
        WhereToCompute(single(readFrame(videoObj))/255, isGPU), ...
        1/downSampleFactor);
    
    for i=1:numberOfStrips
        
        % get current strip
        currentStrip = ...
            currentFrame(stripLocations(i):stripLocations(i)+stripHeight-1,...
            1:stripWidth);
        
        switch method
            case 'fft'
                % subtract the mean, compute energy, and fft
                currentStripZero = currentStrip - mean(currentStrip(:));
                currentStripEnergy = sqrt(sum(currentStripZero(:).^2));
                ft = fft2(currentStripZero, cm, cn);

                % compute the normalized xcorr
                c = ifft2(conj(ft).*(fr)) .* ieuv / currentStripEnergy;
                
                xAdjust = -1;
                yAdjust = -1;
                
            case 'matlab'
                % MATLAB's method
                c = normxcorr2(currentStrip, refFrame);
                xAdjust = stripWidth;
                yAdjust = stripHeight;
                
            otherwise
        end
        
        % find strip location and corresponding shift (movement)
        [ypeak, xpeak] = find(c==max(c(:)));
        yoffSet = ypeak - yAdjust;
        xoffSet = xpeak - xAdjust;
        
        % increment the sample counter and save the position shift
        sampleCounter = sampleCounter + 1;
        xPos(sampleCounter) = xoffSet;
        yPos(sampleCounter) = yoffSet - stripLocations(i);

        % visualize the template matching
        if isVisualize == 2
            figure(1);
            cla;
            imagesc(refFrame);
            colormap(gray);
            axis image;
            hold on;
            drawrectangle(gca, ...
                'Position',[xoffSet+1, yoffSet+1, stripWidth, stripHeight],...
                'Color',[1 0 0],'FaceAlpha',0.1);
            axis image
        end
        

    end
    
end
elapsedTime = toc(t0);

fprintf('Elapsed time: %.4f seconds \nTime spent per frame: %.4f\nTime spent per strip: %.4f\n\n',...
    elapsedTime, elapsedTime/videoObj.FrameRate*videoObj.Duration, ...
    elapsedTime / length(xPos));

% rescale if downsampled
xPos = xPos * downSampleFactor;
yPos = yPos * downSampleFactor;


if isGPU
    xPos = gather(xPos);
    yPos = gather(yPos);
end

% plot
if isVisualize
    figure;
    plot(xPos,'.');
    hold on;
    plot(yPos,'.');
    xlabel('time (n)')
    ylabel('position (px)')
    ylim([-100 100])
end







