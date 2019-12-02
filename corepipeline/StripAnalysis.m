function [position, timeSec, rawPosition, statsStruct] = ...
    StripAnalysis(inputVideo, params)
%STRIP ANALYSIS Extract eye movements in units of pixels.
%   Cross-correlation of horizontal strips with a pre-defined
%   reference frame.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is the path to the video or a matrix representation of the
%   video that is already loaded into memory.
%
%   |referenceFrame| is the path to the reference frame or a matrix representation of the
%   reference frame.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite                       : set to true to overwrite existing files.
%                                     Set to false to abort the function call if the
%                                     files already exist. (default false)
%   enableVerbosity                 : set to true to report back plots during execution.
%                                     (default false)
%   referenceFrame                  : can be a scalar indicating frame number
%                                     within the video, a full file path to a
%                                     video, a 2D array. (default 1)
%   badFrames                       : vector containing the frame numbers of
%                                     the blink frames. (default [])
%   stripHeight                     : strip height to be used for strip
%                                     analysis in pixels. (default 11)
%   samplingRate                    : sampling rate of the video in Hz.
%                                     (default 540)
%   minimumPeakThreshold            : the minimum value above which a peak
%                                     needs to be in order to be considered 
%                                     a valid correlation. (this applies
%                                     regardless of enableGaussianFiltering)
%                                     (default 0.3)
%   adaptiveSearch                  : set to true to perform search on
%                                     scaled down versions first in order
%                                     to potentially improve computation
%                                     time. (default true)
%   searchWindowHeight              : the height of the search window to be
%                                     used for adaptive search in pixels.
%                                     (relevant only when adaptiveSearch is
%                                     true) (default 79)
%   lookBackTime                    : the amount of time in ms to look back
%                                     on when predicting velocity in adaptive 
%                                     search (relevant only when adaptiveSearch 
%                                     is true) (default 10)
%   enableSubpixelInterpolation     : set to true to estimate peak
%                                     coordinates to a subpixel precision
%                                     through interpolation. (default false)
%   axesHandles                     : axes handle for giving feedback. if not
%                                     provided, new figures are created.
%                                     (relevant only when enableVerbosity is true)
%   corrMethod                      : method to use for cross-correlation.
%                                     you can choose from 'normxcorr' for
%                                     matlab's built-in normxcorr2, 'mex'
%                                     for opencv's correlation, or 'fft'
%                                     for our custom-implemented fast
%                                     correlation method. 'cuda' is placed
%                                     but not implemented yet (default
%                                     'mex')
%   downSampleFactor                : If > 1, the reference frame
%                                     and strips are shrunk by that factor
%                                     prior to correlation. If < 1, every 
%                                     other pixel of the reference frame
%                                     is kept (in a checkerboard-like 
%                                     pattern). (default 1) (experimental)
%   enableGaussianFiltering         : set to true to enable Gaussian filtering. 
%                                     Set to false to disable. (default
%                                     false) (experimental, useful for
%                                     noisy videos, e.g., videos from AMD patients)
%   maximumSD                       : maximum standard deviation allowed when
%                                     a gaussian is fitted around the 
%                                     identified peak--strips with positions
%                                     that have a standard deviation >
%                                     maximumSD will be discarded.
%                                     (relevant only when
%                                     enableGaussianFiltering is true)
%                                     (default 10)
%   SDWindowSize                    : the size of the window to use when
%                                     fitting the gaussian for maximumSD
%                                     in pixels. (relevant only when
%                                     enableGaussianFiltering is true)
%                                     (default 25)
%   neighborhoodSize                : the length of one of the sides of the
%                                     neighborhood area over which
%                                     interpolation is to be performed over
%                                     in pixels. (default 5)
%   subpixelDepth                   : in octaves, the scaling of the desired level of
%                                     subpixel depth. (default 2,i.e., 2^-2 = 0.25px)
%   trim                            : 1x2 array. number of pixels removed from top and
%                                     bottom of the frame if video was
%                                     processed in TrimVideo. (default [0
%                                     0] -- [top, bottom])
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'tslo.avi';
%       params.overwrite = true;
%       params.enableVerbosity = false;
%       params.stripHeight = 11;
%       params.samplingRate = 540;
%       params.enableGaussianFiltering = false;
%       params.minimumPeakThreshold = 0.3;
%       params.adaptiveSearch = true;
%       params.enableSubpixelInterpolation = true;
%       params.subpixelInterpolationParameters.neighborhoodSize = 7;
%       params.subpixelInterpolationParameters.subpixelDepth = 3;
%       
%       [position, timeSec, rawPosition, statsStruct] = ...
%           StripAnalysis(inputVideo, params);

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

[~,callerStr] = fileparts(mfilename);

% logical params
overwrite               = ValidateArgument(params,'overwrite',false,@islogical,callerStr);
enableGPU               = ValidateArgument(params,'enableGPU',false,@islogical,callerStr);
enableVerbosity         = ValidateArgument(params,'enableVerbosity',false,@islogical,callerStr);
enableSubpixelInterp    = ValidateArgument(params,'enableSubpixelInterp',false,@islogical,callerStr);
enableGaussianFiltering = ValidateArgument(params,'enableGaussianFiltering',false,@islogical,callerStr);
corrMethod              = ValidateArgument(params,'corrMethod','mex',@(x) any(contains({'mex','normxcorr','fft','cuda'},x)),callerStr);
peakValueThreshold      = ValidateArgument(params,'peakValueThreshold',0.6,@(x) isscalar(x) & (x>=0) & (x<=1),callerStr);
badFrames               = ValidateArgument(params,'badFrames',false,@(x) all(logical(x)),callerStr);
stripHeight             = ValidateArgument(params,'stripHeight',11,@IsNaturalNumber,callerStr);
samplingRate            = ValidateArgument(params,'samplingRate',540,@IsNaturalNumber,callerStr);
minimumPeakThreshold    = ValidateArgument(params,'minimumPeakThreshold',0.3,@IsNonNegativeRealNumber,callerStr);
adaptiveSearch          = ValidateArgument(params,'adaptiveSearch',true,@islogical,callerStr);
searchWindowHeight      = ValidateArgument(params,'searchWindowHeight',79,@IsPositiveInteger,callerStr);
lookBackTime            = ValidateArgument(params,'lookBackTime',10,@(x) IsPositiveRealNumber(x) & (x>=2),callerStr);
frameRate               = ValidateArgument(params,'frameRate',30,@IsPositiveRealNumber,callerStr);
downSampleFactor        = ValidateArgument(params,'downSampleFactor',1,@IsPositiveInteger,callerStr);
axesHandles             = ValidateArgument(params,'axesHandles',nan,@(x) isnan(x) | all(ishandle(x)),callerStr);
neighborhoodSize        = ValidateArgument(params,'neighborhoodSize',5,@IsPositiveInteger,callerStr);
subpixelDepth           = ValidateArgument(params,'subpixelDepth',2,@IsPositiveInteger,callerStr);
maximumSD               = ValidateArgument(params,'maximumSD',10,@IsPositiveRealNumber,callerStr);
SDWindowSize            = ValidateArgument(params,'SDWindowSize',25,@IsPositiveRealNumber,callerStr);
trim                    = ValidateArgument(params,'trim',[0 0],@(x) all(IsNaturalNumber(x)) & (length(x)==2),callerStr);
referenceFrame          = ValidateArgument(params,'referenceFrame',1,...
    @(x) isscalar(x) | ischar(x) | (isnumeric(x) & size(x,1)>1 & size(x,2)>1),callerStr);

% check if CUDA enabled GPU exists
if enableGPU && (gpuDeviceCount < 1)
    enableGPU = false;
    RevasWarning('No supported GPU available. StripAnalysis is reverting back to CPU', params);
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputVideo, 'usefultraces', params.samplingRate);

    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~overwrite
        RevasMessage(['StripAnalysis() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        RevasMessage('StripAnalysis() is returning results from existing file.',params); 
        
        % try loading existing file contents
        load(outputFilePath,'position', 'timeSec', 'rawPosition', 'statsStruct');
        return;
    else
        RevasMessage(['StripAnalysis() is proceeding and overwriting an existing file. (' outputFilePath ')'], params);  
    end
end


%% Create a reader object if needed and get some info on video

if writeResult
    reader = VideoReader(inputVideo);
    frameRate = reader.FrameRate;
    width = reader.Width;
    height = reader.Height;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    [height, width, numberOfFrames] = size(inputVideo);
end


%% Handle referenceFrame separately 

% if referenceFrame is a scalar, it refers to the frame number within the
% video that can be used as the initial reference frame
if isscalar(referenceFrame)
    if writeResult
        referenceFrame = ReadSpecificFrame(reader,referenceFrame);
        reader.CurrentTime = 0; % rewind back current time
    else
        referenceFrame = inputVideo(:,:,referenceFrame);
    end
end

% if it's a path, read the image from the path
if ischar(referenceFrame) 
    referenceFrame = imread(referenceFrame);
end

% if larger than 1, apply downsampling to reduce size of the images.
if downSampleFactor > 1
    referenceFrame = imresize(referenceFrame,1/downSampleFactor);
    height = round(height / downSampleFactor);
    stripHeight = max(1,round(stripHeight / downSampleFactor));
    trim = round(trim / downSampleFactor);
end

% at this point, referenceFrame should be a 2D array.
assert(ismatrix(referenceFrame));


%% Prepare variables before the big loop

[refHeight, refWidth] = size(referenceFrame);
stripsPerFrame = round(samplingRate / frameRate);
rowNumbers = round(linspace(1,(height - stripHeight),stripsPerFrame));
numberOfStrips = stripsPerFrame * numberOfFrames;

% two columns for horizontal and vertical movements
rawPosition = nan(numberOfStrips, 2);
position = nan(numberOfStrips, 2);

% arrays for peak values
peakValueArray = zeros(numberOfStrips, 1);

% Populate time array
dtPerScanLine = 1/((height + sum(trim)) * frameRate);
timeSec = dtPerScanLine * ...
    reshape(trim(1) + rowNumbers' + ...
    (0:(numberOfFrames-1)) * (height + sum(trim)),numberOfStrips,1);

% if gpu is enabled, move reference frame to gpu memory
if enableGPU && ~contains(corrMethod,'cuda')
    referenceFrame = gpuArray(referenceFrame);
end

% Variables for fft corrmethod
if contains(corrMethod, 'fft')
    cache = struct;
end

% Variables for adaptive search:
if adaptiveSearch
    
    % within a single frame, assuming no vertical eye motion, strips should
    % be at the following distance from each other on the reference.
    verticalIncrement = median(diff(rowNumbers));
    
    % remember the last few velocities, according to lookBackTime (no fewer than 2 samples)
    historyCapacity = floor(lookBackTime / 1000 * samplingRate);
    
    if historyCapacity < 2
        historyCapacity = 2;
        RevasMessage(['StripAnalysis(): setting historyCapacity to 2. (' outputFilePath ')'], params);  
    end
    
    movementHistory = nan(historyCapacity,1);
    
    % set number of attempts. After first attempt with searchWindowHeight,
    % we'll double the window height. 
    numOfAttempts = 3; 
else
    numOfAttempts = 1;
end


%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end
isSetView = true;


%% Extract motion by template matching strips to reference frame

% loop across frames
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
        
        % if GPU is enabled, transfer the frame to GPU memory
        if enableGPU
            frame = gpuArray(frame);
        end
        
        % downsample if requested.
        if downSampleFactor > 1
            frame = imresize(frame, 1/downSampleFactor);
        end
        
        % loop across strips within current frame
        for sn = 1:stripsPerFrame
            
            thisSample = stripsPerFrame * (fr-1) + sn;
            thisStrip = frame(rowNumbers(sr) : (rowNumbers(sr)+stripHeight-1),:);
            
            for attempt = 1:numOfAttempts
                
                % if adaptive search is on, get the most likely place in the
                % frame that will contain the current strip
                if adaptiveSearch
                    % for first sample do full cross-corr.
                    if sn == 1 && fr == 1
                        rowStart = 1;
                        rowEnd = refHeight;
                    else
                        % Take the last peak position in the correlation map,
                        % add one inter-strip vertical displacement, average
                        % displacement in the moving buffer, and (if this is
                        % the first strip of a frame other than first frame)
                        % subtract one frame-stripHeight.
                        loc =  rawPosition(thisSample-1,2) + ...
                            (sn ~= 1) * verticalIncrement + ...
                            nansum([0 nanmean(movementHistory)]) - ...
                            (sn == 1) * (height - stripHeight);
                        thisWindowHeight = searchWindowHeight * 2^(attempt-1); 
                        rowStart = max(1, loc - floor(thisWindowHeight/2));
                        rowEnd = max(refHeight, loc + floor(thisWindowHeight/2));
                    end
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % locate the strip
                switch corrMethod
                    case 'mex'
                        if enableGPU
                            correlationMap = matchTemplateOCV_GPU(thisStrip, referenceFrame(rowStart:rowEnd,:)); 
                            [xPeak, yPeak, peakValue] = FindPeak(correlationMap, enableGPU);
                        else
                            [correlationMap,xPeak,yPeak,peakValue] = ...
                                matchTemplateOCV(thisStrip, referenceFrame(rowStart:rowEnd,:)); 
                        end

                    case 'normxcorr'
                        correlationMap = normxcorr2(thisStrip, referenceFrame(rowStart:rowEnd,:)); 
                        [xPeak, yPeak, peakValue] = FindPeak(correlationMap, enableGPU);

                    case 'fft'
                        [correlationMap, cache,xPeak,yPeak,peakValue] = ...
                            FastStripCorrelation(thisStrip, referenceFrame(rowStart:rowEnd,:), cache, enableGPU);

                    case 'cuda'
                        % TO-DO

                    otherwise
                        error('StripAnalysis: unknown corrMethod.');
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                % visualization, if requested.
                if enableVerbosity
                    
                end
                
                
                % check if criterion is satisfied, if so break the attempt
                % loop and continue to the next strip
                if peakValue >= minimumPeakThreshold
                    break;
                end
            end
            
            % update the traces/stats
            peakValueArray(thisSample) = peakValue;
            rawPosition(thisSample,:) = [xPeak (yPeak+rowStart-1)];
            
            if adaptiveSearch
                movementHistory = circshift(movementHistory,1,1);
                movementHistory(1) = diff(rawPosition(end-1:end,2));
            end
            
        end % end of frame
    end % abortTriggered
end % end of video











































%% Normalized cross-correlate each strip

        
        % Show surface plot for this correlation if verbosity enabled
        if params.enableVerbosity
            if params.enableGPU
                correlationMap = gather(correlationMap);
            end
            if isfield(params, 'axesHandles')
                axes(params.axesHandles(1)); %#ok<*LAXES>
                cla;
                colormap(params.axesHandles(1), 'default');
                if isSetView
                    view(3)
                    isSetView = false;
                end
            else
                figure(1);
                cla;
            end
            
            [surfX,surfY] = meshgrid( ...
                1 : size(correlationMap, 2), ...
                upperBound : upperBound + size(correlationMap,1) - 1);
            surf(surfX, surfY, correlationMap, 'linestyle', 'none');
            title([num2str(stripNumber) ' out of ' num2str(numberOfStrips)]);
            xlim([1 size(correlationMap, 2)]);
            ylim([upperBound upperBound + size(correlationMap,1) - 1]);
            zlim([-1 1]);
            xlabel('');
            ylabel('');
            legend('off');
            
            % Mark the identified peak on the plot with an arrow.
            text(double(xPeak), double(yPeak), double(peakValue), ...
                '\downarrow', 'Color', 'red', ...
                'FontSize', 20, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

            drawnow;  
        end

        % If verbosity is enabled, also show eye trace plot with points
        % being plotted as they become available.
        if params.enableVerbosity

            % Adjust for padding offsets added by normalized cross-correlation.
            % If we enable verbosity and demand that we plot the points as we
            % go, then adjustments must be made here in order for the plot to
            % be interpretable.
            % Therefore, we will only perform these same operations after the
            % loop to take advantage of vectorization only if they are not
            % performed here, namely, if verbosity is not enabled and this
            % if statement does not execute.
            if ~isequal(params.corrMethod, 'fft')
                rawPosition(stripNumber,2) = ...
                    rawPosition(stripNumber,2) - (params.stripHeight - 1);
                rawPosition(stripNumber,1) = ...
                    rawPosition(stripNumber,1) - (params.stripWidth - 1);
            end
            
            % Only scale up if downSampleFactor is > 1, since this means it
            % was shrunk during correlation.
            % If downSampleFactor was < 1, the images were thrown against a
            % bernoulli mask, but remained the same overall dimension.
            if params.downSampleFactor > 1
               rawPosition(stripNumber, :) = rawPosition(stripNumber, :) .* params.downSampleFactor; 
            end

            % Negate eye position traces to flip directions.
            rawPosition(stripNumber,:) = -rawPosition(stripNumber,:);

            if isfield(params, 'axesHandles')
                axes(params.axesHandles(2));
                cla;
                colormap(params.axesHandles(2), 'default');
            else
                figure(2);
                cla;
            end
            
            plot(timeSec, rawPosition(:,1),'-r','linewidth',2); hold on;
            plot(timeSec, rawPosition(:,2),'-b','linewidth',2);
            title('Raw Eye Position');
            xlabel('time (sec)');
            ylabel('eye position (pixels)');
            legend('show');
            legend('horizontal', 'vertical');
            set(gca,'fontsize',14);
            
%             figure(3);
%             cla;
%             plot(timeArray,peakValueArray,'-','color',[.2 .8 .2],'linewidth',2); hold on;
%             plot(timeArray,1 - (secondPeakValueArray./peakValueArray),'-','color',[.6 .2 .6],'linewidth',2);
%             legend('show');
%             legend('Peak','1-PeakRatio');
%             ylabel('quality')
%             set(gca,'fontsize',14);
%             xlabel('time (sec)')
            
        end


%% Adjust for padding offsets added by normalized cross-correlation.
% Do this after the loop to take advantage of vectorization
% Only run this section if verbosity was not enabled. If verbosity was
% enabled, then these operations were already performed for each point
% before it was plotted to the eye traces graph. If verbosity was not
% enabled, then we do it now in order to take advantage of vectorization.
if ~params.enableVerbosity
    if ~isequal(params.corrMethod, 'fft')
        rawPosition(:,2) = ...
            rawPosition(:,2) - (params.stripHeight - 1);
        rawPosition(:,1) = ...
            rawPosition(:,1) - (params.stripWidth - 1);
    end
    
    % Only scale up if downSampleFactor is > 1, since this means it
    % was shrunk during correlation.
    % If downSampleFactor was < 1, the images were thrown against a
    % checkboard mask, but remained the same overall dimension.
    if params.downSampleFactor > 1
       rawPosition = rawPosition .* params.downSampleFactor; 
    end

    % Negate eye position traces to flip directions.
    rawPosition = -rawPosition;
end



%% Plot Useful Eye Traces
if ~abortTriggered && params.enableVerbosity
    if isfield(params, 'axesHandles')
        axes(params.axesHandles(2));
        colormap(params.axesHandles(2), 'gray');
    else
        figure(3);
    end
    plot(timeSec, position);
    title('Useful Eye Position Traces');
    xlabel('Time (sec)');
    ylabel('Eye Position Traces (pixels)');
    legend('show');
    legend('Horizontal Traces', 'Vertical Traces');
    
    if isfield(params, 'axesHandles')
        axes(params.axesHandles(1));
    else
        figure(11);
    end
    cla;
    
    plot(timeSec,peakValueArray,'-','color',[.2 .8 .2],'linewidth',2); hold on;
    plot(timeSec,1 - (secondPeakValueArray./peakValueArray),'-','color',[.6 .2 .6],'linewidth',2);
    legend('show');
    legend('Peak','1-PeakRatio');
    ylabel('quality')
    set(gca,'fontsize',14);
    xlabel('time (sec)')
    title('Sample quality');
    ylim([0 1])
    view(2);
end

%% Plot stimuli on reference frame
if ~abortTriggered && params.enableVerbosity
    if isfield(params, 'axesHandles')
        axes(params.axesHandles(3));
        colormap(params.axesHandles(3), 'gray');
    else
        figure(4);
    end
    
    imshow(referenceFrame);
    hold on;
    
    center = fliplr(size(referenceFrame)/2);
    positionsToBePlotted = repmat(center, length(position),1) + position;
    
    scatter(positionsToBePlotted(:,1), positionsToBePlotted(:,2), 'y', 'o' , 'filled');
    hold off;
end

%% Save to output mat file

if writeResult && ~abortTriggered
    
    try
        params = rmfield(params,'axesHandles'); 
        params = rmfield(params,'commandWindowHandle'); 
    catch
    end
    
    % Save under file labeled 'final'.
    if writeResult
        eyePositionTraces = position; 
        peakRatios = statisticsStructure.peakRatios;
        save(outputFilePath, 'eyePositionTraces', 'rawEyePositionTraces', ...
            'timeArray', 'params', 'referenceFramePath', ...
            'peakRatios');
    end
end


