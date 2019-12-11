function [position, timeSec, rawPosition, peakValueArray, varargout] = ...
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
%   |referenceFrame| is the path to the reference frame or a matrix
%   representation of the reference frame.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%
%   overwrite         : set to true to overwrite existing files. Set to 
%                       false to abort the function call if the files
%                       already exist. (default false)
%   enableVerbosity   : set to true to report back plots during execution. (
%                       default false)
%   enableGPU         : a logical. if set to true, use GPU. (works for 
%                       'mex' method only.
%   enableSubpixel    : set to true to estimate peak coordinates to a 
%                       subpixel precision through interpolation. (default
%                       false)
%   corrMethod        : method to use for cross-correlation. you can 
%                       choose from 'normxcorr' for matlab's built-in
%                       normxcorr2, 'mex' for opencv's correlation, or 
%                       'fft' for our custom-implemented fast
%                       correlation method. 'cuda' is placed but not 
%                       implemented yet (default 'mex').
%   referenceFrame    : can be a scalar indicating frame number within the
%                       video, a full file path to a video, a 2D array.
%                       (default 1)
%   badFrames         : vector containing the frame numbers of the blink 
%                       frames. (default [])
%   stripHeight       : strip height to be used for strip analysis in 
%                       pixels. (default 11)
%   samplingRate      : sampling rate of the video in Hz. (default 540)
%   minPeakThreshold  : the minimum value above which a peak needs to be 
%                       in order to be considered a valid correlation.
%                       (default 0.3)
%   adaptiveSearch    : set to true to perform search on scaled down 
%                       versions first in order to potentially improve
%                       computation time. (default true)
%   searchWindowHeight: the height of the search window to be used for 
%                       adaptive search in pixels. (relevant only when
%                       adaptiveSearch is true) (default 79)
%   lookBackTime      : the amount of time in ms to look back on when 
%                       predicting velocity in adaptive search (relevant
%                       only when adaptiveSearch is true) (default 10)
%   frameRate         : Frame rate of input video in Hz. needed to specify 
%                       explicitly if the inputVideo is a 3D array rather
%                       than a path to a video.
%   axesHandles       : axes handle for giving feedback. if not provided, 
%                       new figures are created. (relevant only when
%                       enableVerbosity is true)
%   neighborhoodSize  : the length of one of the sides of the neighborhood 
%                       area over which interpolation is to be performed
%                       over in pixels. (default 5)
%   subpixelDepth     : in octaves, the scaling of the desired level of 
%                       subpixel depth. (default 2,i.e., 2^-2 = 0.25px)
%   trim              : 1x2 array. number of pixels removed from top and 
%                       bottom of the frame if video was processed in
%                       TrimVideo. (default [0 0] -- [top, bottom]).
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |position| is a Nx2 array of useful eye motions. Useful in the sense
%   that peak values are above minPeakThreshold criterion.
%
%   |timeSec| is a Nx1 array of time in seconds.
%
%   |rawPosition| is a Nx2 array of raw eye motion.
%
%   |peakValueArray| is a Nx1 array of peaks of cross-correlation maps.
%
%   |varargout| is a variable output argument holder. Used to return the 
%   'params' structure. 
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'tslo.avi';
%       params.overwrite = true;
%       params.enableVerbosity = false;
%       params.stripHeight = 11;
%       params.samplingRate = 540;
%       params.minimumPeakThreshold = 0.3;
%       
%       [position, timeSec, rawPosition] = StripAnalysis(inputVideo, params);
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


%% Handle GPU 

% check if CUDA enabled GPU exists
if params.enableGPU && (gpuDeviceCount < 1)
    params.enableGPU = false;
    RevasWarning('No supported GPU available. StripAnalysis is reverting back to CPU', params);
end


%% Handle verbosity 

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'units','normalized','outerposition',[0.16 0.053 0.67 0.51]);
    params.axesHandles(1) = subplot(2,3,[1 2 4 5]);
    params.axesHandles(2) = subplot(2,3,3);
    params.axesHandles(3) = subplot(2,3,6);
    for i=1:3
        cla(params.axesHandles(i));
    end
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputVideo, 'usefultraces', params.samplingRate);
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['StripAnalysis() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        RevasMessage('StripAnalysis() is returning results from existing file.',params); 
        
        % try loading existing file contents
        load(outputFilePath,'position', 'timeSec', 'rawPosition', 'peakValueArray','params');
        if nargout > 4
            varargout{1} = params;
        end
        
        return;
    else
        RevasMessage(['StripAnalysis() is proceeding and overwriting an existing file. (' outputFilePath ')'], params);  
    end
end


%% Create a reader object if needed and get some info on video

if writeResult
    reader = VideoReader(inputVideo);
    params.frameRate = reader.FrameRate;
    width = reader.Width; 
    height = reader.Height;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    [height, width, numberOfFrames] = size(inputVideo); 
end


%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);


%% Handle referenceFrame separately 

% if referenceFrame is a scalar, it refers to the frame number within the
% video that can be used as the initial reference frame
if isscalar(params.referenceFrame)
    if writeResult
        params.referenceFrame = ReadSpecificFrame(reader,params.referenceFrame);
        reader.CurrentTime = 0; % rewind back current time
    else
        params.referenceFrame = inputVideo(:,:,params.referenceFrame);
    end
end

% if it's a path, read the image from the path
if ischar(params.referenceFrame) 
    params.referenceFrame = imread(params.referenceFrame);
end

% at this point, referenceFrame should be a 2D array.
assert(ismatrix(params.referenceFrame));

%% Prepare variables before the big loop

[refHeight, refWidth] = size(params.referenceFrame); %#ok<ASGLU>
stripsPerFrame = round(params.samplingRate / params.frameRate);
rowNumbers = round(linspace(params.stripHeight,(height - 2*params.stripHeight),stripsPerFrame));
numberOfStrips = stripsPerFrame * numberOfFrames;

% two columns for horizontal and vertical movements
rawPosition = nan(numberOfStrips, 2);
peakPosition = nan(numberOfStrips, 2);
position = nan(numberOfStrips, 2);

% arrays for peak values
peakValueArray = zeros(numberOfStrips, 1);

% Populate time array
dtPerScanLine = 1/((height + sum(params.trim)) * params.frameRate);
if length(params.badFrames) > numberOfFrames
    absoluteFrameNo = (find(~params.badFrames)-1)';
else
    absoluteFrameNo = (0:(numberOfFrames-1));
end
timeSec = dtPerScanLine * ...
    reshape(params.trim(1) + rowNumbers' + ...
    absoluteFrameNo * (height + sum(params.trim)),numberOfStrips,1);

% if gpu is enabled, move reference frame to gpu memory
if params.enableGPU && ~contains(params.corrMethod,'cuda')
    params.referenceFrame = gpuArray(params.referenceFrame);
end

% Variable for fft corrmethod
cache = struct;


% Variables for adaptive search:
if params.adaptiveSearch
    
    % within a single frame, assuming no vertical eye motion, strips should
    % be at the following distance from each other on the reference.
    verticalIncrement = median(diff(rowNumbers));
    
    % remember the last few velocities, according to lookBackTime (no fewer than 2 samples)
    historyCapacity = floor(params.lookBackTime / 1000 * params.samplingRate);
    
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


%% Extract motion by template matching strips to reference frame

% the big loop. :)

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
        if params.skipFrame(fr)
            continue;
        end
        
        % if GPU is enabled, transfer the frame to GPU memory
        if params.enableGPU
            frame = gpuArray(frame);
        end
        
        % loop across strips within current frame
        for sn = 1:stripsPerFrame
            
            % get current strip
            thisSample = stripsPerFrame * (fr-1) + sn;
            thisStrip = frame(rowNumbers(sn) : (rowNumbers(sn)+params.stripHeight-1),:);
            
            for attempt = 1:numOfAttempts
                
                % if adaptive search is on, get the most likely place in the
                % frame that will contain the current strip
                if params.adaptiveSearch
                    
                    % for first sample do full cross-corr.
                    if sn == 1 && fr == 1
                        params.rowStart = 1;
                        params.rowEnd = refHeight;
                        
                    else  
                    % for the rest of samples, predict the location of peak
                    % location based on history of eye motion.
                    
                        % Take the last peak position in the correlation map,
                        % add one inter-strip vertical displacement, average
                        % displacement in the moving buffer, and (if this is
                        % the first strip of a frame other than first frame)
                        % subtract one frame-stripHeight.
                        loc =  mod((sn ~= 1) * peakPosition(thisSample-1,2) + ...
                            (sn ~= 1) * verticalIncrement + ...
                            nansum([0 nanmedian(movementHistory)]) , refHeight);
                        thisWindowHeight = params.searchWindowHeight * 2^(attempt-1); 
                        
                        % check for out of referenceFrame bounds
                        params.rowStart = max(1, loc - floor(thisWindowHeight/2));
                        params.rowEnd = min(refHeight, loc + floor(thisWindowHeight/2));
                        
                        
                        % check if size is smaller than desired. if
                        % so, enlarge the reference.
                        if params.rowEnd - params.rowStart <= thisWindowHeight
                            if params.rowEnd == height
                                params.rowStart = params.rowEnd - thisWindowHeight + 1;
                            end
                            if params.rowStart == 1
                                params.rowEnd = params.rowStart + thisWindowHeight - 1;
                            end
                        end
                    end
                else
                    params.rowStart = 1;
                    params.rowEnd = refHeight;
                end % end of adaptive
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % locate the strip
                [correlationMap, xPeak, yPeak, peakValue, cache] = ...
                    LocateStrip(thisStrip,params,cache);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                % visualization, if requested.
                if params.enableVerbosity > 1
                    
                    % show cross-correlation output
                    if params.enableGPU
                        correlationMap = gather(correlationMap);
                    end
                    
                    axes(params.axesHandles(1)); %#ok<LAXES>
                    imagesc(correlationMap);
                    colorbar(params.axesHandles(1));
                    caxis(params.axesHandles(1),[-1 1]);
                    hold(params.axesHandles(1),'on');
                    scatter(params.axesHandles(1),xPeak,yPeak,100,'r');
                    hold(params.axesHandles(1),'off');
                    
                    drawnow; % maybe needed in GUI mode.
                end
                
                
                % check if criterion is satisfied, if so break the attempt
                % loop and continue to the next strip. If adaptive search
                % is off, this is ineffective (since 'numOfAttempts' is set
                % to 1 in that case).
                if peakValue >= params.minPeakThreshold
                    break;
                end
            end % end of attempts
            
            % if subpixel estimation is requested
            if params.enableSubpixel
                [xPeak, yPeak, peakValue] = ...
                    Interpolation2D(correlationMap, xPeak, yPeak, ...
                        params.neighborhoodSize, params.subpixelDepth, [], params.enableGPU);
            end
            
            % update the traces/stats
            peakValueArray(thisSample) = peakValue;
            peakPosition(thisSample,:) = [xPeak, yPeak - params.stripHeight + params.rowStart - 1];
            rawPosition(thisSample,:) = [xPeak-width ...
                (yPeak - params.stripHeight - rowNumbers(sn) + params.rowStart - 1)];
            
            % if adaptive search is enabled, circularly shift the buffer to
            % update the last N positions.
            if params.adaptiveSearch
                movementHistory = circshift(movementHistory,1,1);
                if thisSample > 1
                    movementHistory(1) = diff(rawPosition(thisSample-1:thisSample,2));
                end
            end
            
            % plot peak values and raw traces
            if params.enableVerbosity > 1
                % show peak values
                plot(params.axesHandles(2),timeSec,peakValueArray,'-','linewidth',2); 
                hold(params.axesHandles(2),'on');
                plot(params.axesHandles(2),timeSec([1 end]),params.minPeakThreshold*ones(1,2),'--','color',.7*[1 1 1],'linewidth',2);
                set(params.axesHandles(2),'fontsize',14);
                xlabel(params.axesHandles(2),'time (sec)');
                ylabel(params.axesHandles(2),'peak value');
                ylim(params.axesHandles(2),[0 1]);
                xlim(params.axesHandles(2),[0 max(timeSec)]);
                hold(params.axesHandles(2),'off');
                grid(params.axesHandles(2),'on');

                % show raw output traces
                plot(params.axesHandles(3),timeSec,rawPosition,'-','linewidth',2);
                set(params.axesHandles(3),'fontsize',14);
                xlabel(params.axesHandles(3),'time (sec)');
                ylabel(params.axesHandles(3),'position (px)');
                legend(params.axesHandles(3),{'hor','ver'});
                yMin = max([-100, prctile(rawPosition,5,'all')-10]);
                yMax = min([100, prctile(rawPosition,95,'all')+10]);
                ylim(params.axesHandles(3),[yMin yMax]);
                xlim(params.axesHandles(3),[0 max(timeSec)]);
                hold(params.axesHandles(3),'off');
                grid(params.axesHandles(3),'on');
                
                drawnow; % maybe needed in GUI mode.
            end % en of plot
            
        end % end of frame
    end % abortTriggered
end % end of video


% good samples
ix = peakValueArray >= params.minPeakThreshold;
position(ix,:) = rawPosition(ix,:);


%% Plot stimuli on reference frame
if ~abortTriggered && params.enableVerbosity 
    
    % compute position on the reference frame
    center = fliplr(size(params.referenceFrame)/2);
    positionsToBePlotted = repmat(center, size(position,1),1) + position;
    
    % plot stimulus motion on the retina
    axes(params.axesHandles(1));
    imagesc(params.referenceFrame);
    colormap(params.axesHandles(1),gray(256));
    hold(params.axesHandles(1),'on');
    scatter(params.axesHandles(1),positionsToBePlotted(:,1), positionsToBePlotted(:,2), 10, parula(length(timeSec)), 'o' , 'filled');
    hold(params.axesHandles(1),'off');
    cb = colorbar(params.axesHandles(1));
    cb.Label.String = 'time (sec)';
    cb.FontSize = 14;
    axis(params.axesHandles(1),'image')
    xlim(params.axesHandles(1),[1 size(params.referenceFrame,2)])
    ylim(params.axesHandles(1),[1 size(params.referenceFrame,1)])
    
    % show peak values
    plot(params.axesHandles(2),timeSec,peakValueArray,'-','linewidth',2); 
    hold(params.axesHandles(2),'on');
    plot(params.axesHandles(2),timeSec([1 end]),params.minPeakThreshold*ones(1,2),'--','color',.7*[1 1 1],'linewidth',2);
    set(params.axesHandles(2),'fontsize',14);
    xlabel(params.axesHandles(2),'time (sec)');
    ylabel(params.axesHandles(2),'peak value');
    ylim(params.axesHandles(2),[0 1]);
    xlim(params.axesHandles(2),[0 max(timeSec)]);
    hold(params.axesHandles(2),'off');
    grid(params.axesHandles(2),'on');% all peak values already plotted, so we skip.

    % plot useful position traces.
    plot(params.axesHandles(3),timeSec,position,'-','linewidth',2);
    set(params.axesHandles(3),'fontsize',14);
    xlabel(params.axesHandles(3),'time (sec)');
    ylabel(params.axesHandles(3),'position (px)');
    legend(params.axesHandles(3),{'hor','ver'});
    yMin = max([-100, prctile(position,5,'all')-10]);
    yMax = min([100, prctile(position,95,'all')+10]);
    ylim(params.axesHandles(3),[yMin yMax]);
    xlim(params.axesHandles(3),[0 max(timeSec)]);
    hold(params.axesHandles(3),'off');
    grid(params.axesHandles(3),'on');
end


%% Save to output mat file

if writeResult && ~abortTriggered
    
    % saving axes handles or handle to command window creates issues. so
    % remove those fields here.
    try
        params = rmfield(params,'axesHandles'); 
        params = rmfield(params,'commandWindowHandle'); 
    catch
    end
    
    % Save under file labeled 'final'.
    if writeResult
        save(outputFilePath, 'position', 'rawPosition', 'timeSec', 'params');
    end
end


%% return the params structure if requested

if nargout > 4
    varargout{1} = params;
end



