function [xPos, yPos, t] = FastStripAnalysis(pathToVideo, refFrame, method, isGPU, isVisualize)

if nargin < 1 || isempty(pathToVideo)
    error('We need a video path!');
end

if nargin < 2 || isempty(refFrame)
    
    % try locating based on video path
    try
    load([pathToVideo(1:end-4) '_refframe.mat'], 'refFrame');
    catch
        disp('''_refframe.mat'' does not exist.')
        error('We need a reference frame');
    end
end

if nargin < 3 || isempty(method)
    method = 'fft';
end

if nargin < 4 || isempty(isGPU)
    isGPU = false;
end

if nargin < 5 || isempty(isVisualize)
    isVisualize = 0;
end

switch method
    case {'fft','matlab'}
        refFrame = ...
            WhereToCompute(single(refFrame)/255, isGPU);
    case 'opencv'
        refFrame = ...
            WhereToCompute((refFrame), isGPU);
end

% create a video reader object
videoObj = VideoReader(pathToVideo);
startTime = videoObj.CurrentTime;


% rewind back to the beginning of the video
videoObj.CurrentTime = startTime;

% define strip parameters
stripHeight = ceil(11);
stripWidth = ceil(videoObj.Width);
numberOfStrips = 18;
delta = 1;
stripLocations = round(linspace(delta, ...
    videoObj.Height - stripHeight + 1, numberOfStrips))';

% contruct time array
t = linspace(0,videoObj.Duration,numberOfStrips*videoObj.FrameRate*videoObj.Duration)';
        

if contains(method,'fft')
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
    ieuv = 1./sqrt(circshift((euv), -[stripHeight stripWidth]+1));

    [refFrameHeight, refFrameWidth] = size(refFrame);

    % fft of the reference
    cm = stripHeight + refFrameHeight - 1;
    cn = stripWidth  + refFrameWidth  - 1;
    fr = fft2(refFrame, cm, cn);
end

% preallocate arrays
xPos = WhereToCompute(nan(videoObj.FrameRate * videoObj.Duration * numberOfStrips,1),...
    isGPU);
yPos = xPos;
sampleCounter = 0;

% analyze video
while hasFrame(videoObj)
    
    switch method
        case {'fft','matlab'}
            currentFrame =  ...
                WhereToCompute(single(readFrame(videoObj))/255, isGPU);
        case 'opencv'
            currentFrame =  ...
                WhereToCompute((readFrame(videoObj)), isGPU);
    end
    
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
                
                xAdjust = 0;
                yAdjust = 0;
                
            case 'matlab'
                % MATLAB's method
                c = normxcorr2(currentStrip, refFrame);
                xAdjust = stripWidth;
                yAdjust = stripHeight;
                
            case 'opencv'
                if isGPU
                    c = matchTemplateOCV_GPU(currentStrip, refFrame);
                    xAdjust = 0;
                    yAdjust = 0;
                else
                    c = matchTemplateOCV(currentStrip, refFrame);
                    xAdjust = stripWidth-1;
                    yAdjust = stripHeight-1;
                end
                
                
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


if isGPU
    xPos = gather(xPos);
    yPos = gather(yPos);
end


xPos = -xPos;
yPos = -yPos;

% plot
if isVisualize
    figure(1);
    subplot(1,2,1)
    plot(xPos,'.-');
    hold on; grid on;
    set(gca,'fontsize',14)
    xlabel('time (n)')
    ylabel('position (px)')
    ylim([-100 100])
    subplot(1,2,2)
    plot(yPos,'.-');
    hold on; grid on;
    set(gca,'fontsize',14)
    xlabel('time (n)')
    ylabel('position (px)')
    ylim([-100 100])
end







