function [xPos, yPos] = FastStripAnalysis(pathToVideo, isGPU, isVisualize)

if nargin < 1 || isempty(pathToVideo)
    pathToVideo = '../demo/sample10deg_nostim_gamscaled_bandfilt 1.avi';
end

if nargin < 2 || isempty(isGPU)
    isGPU = false;
end

if nargin < 3 || isempty(isVisualize)
    isVisualize = 0;
end

method = 'fft';
downSampleFactor = 2;

% create a video reader object
videoObj = VideoReader(pathToVideo);
startTime = videoObj.CurrentTime;

% use the first frame as the reference frame
reference = imresize(WhereToCompute(single(readFrame(videoObj))/255, isGPU),...
    1/downSampleFactor);

% rewind back to the beginning of the video
videoObj.CurrentTime = startTime;

% define strip parameters
stripHeight = 8/downSampleFactor;
stripWidth = videoObj.Width/downSampleFactor;
numberOfStrips = 4;
delta = 16;
stripLocations = round(linspace(delta, ...
    videoObj.Height - stripHeight - 1, numberOfStrips));

% precomputed arrays
mask = WhereToCompute(ones(stripHeight, stripWidth,'single'), isGPU);
fuv = conv2(reference,mask);
f2uv = conv2(reference.^2,mask);

% precision of the computations
eps = 10^-6;

% energy of the reference
euv = (f2uv - (fuv.^2)/(stripHeight * stripWidth));
euv(euv == 0) = eps;

% shift, sqrt, and take the reciprocal of euv here for speed up. Note that
% division is more expensive than multiplication.
ieuv = 1./sqrt(circshift(euv, -[stripHeight stripWidth]+1));

% fft of the reference
cm = stripHeight + videoObj.Height/downSampleFactor - 1;
cn = stripWidth  + videoObj.Width/downSampleFactor  - 1;
fr = fft2(reference, cm, cn);

% preallocate arrays
xPos = WhereToCompute(nan(videoObj.FrameRate * videoObj.Duration * numberOfStrips,1),...
    isGPU);
yPos = xPos;
sampleCounter = 0;

% analyze video
t0 = tic;
while hasFrame(videoObj)
    
    currentFrame = WhereToCompute(single(readFrame(videoObj))/255, isGPU);
    
    for i=1:numberOfStrips
        
        % get current strip
        currentStrip = imresize(...
            currentFrame(stripLocations(i):stripLocations(i)+stripHeight-1,...
            1:stripWidth),1/downSampleFactor);
        
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
                c = normxcorr2(currentStrip, reference);
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
            imagesc(reference);
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







