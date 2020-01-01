function [position, timeSec, rawPosition, peakValueArray, varargout] = ...
    StripAnalysis(inputVideo, params)
%[position, timeSec, rawPosition, peakValueArray, varargout] = StripAnalysis(inputVideo, params)
%   
%   Extract eye movements in units of pixels. Cross-correlation of
%   horizontal strips with a pre-defined reference frame.
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
%                       false to abort the function call if the files
%                       already exist. (default false)
%   enableVerbosity   : set to true to report back plots during execution. (
%                       default false)
%   enableGPU         : a logical. if set to true, use GPU. (works for 
%                       'mex' method only.
%   enableReferenceFrameUpdate: Useful for initial run of StripAnalysis for
%                       making a reference frame from raw videos. When
%                       enabled, reference frame is changed on the fly to a
%                       good quality video frame that is closer in time to
%                       the current strip. Does not make much difference
%                       for short videos (e.g., 1sec) but it improves
%                       quality of extraction quite significantly for long
%                       videos (e.g., duration>5sec).
%   goodFrameCriterion: Relevant only when 'enableReferenceFrameUpdate' is
%                       set to true. It represents the proportion of
%                       samples from a single frame whose peak value is
%                       above 'minPeakThreshold' (specified below). (0-1),
%                       defaults to 0.9. E.g., if number of strips per
%                       frame is 20, at least 0.9*20=18 samples must
%                       satisfy the 'minPeakThreshold' for a frame to be
%                       considered as a candidate reference frame.
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
%   stripWidth        : strip width in pixels. default []. When empty,
%                       strip width is set to frame width.
%   samplingRate      : sampling rate of the video in Hz. (default 540)
%   minPeakThreshold  : the minimum value above which a peak needs to be 
%                       in order to be considered a valid correlation.
%                       (default 0.3)
%   maxMotionThreshold: Maximum motion between successive strips.
%                       Specificed in terms of ratio with respect to the
%                       frame height. Value must be between 0 and 1. E.g.,
%                       0.05 for a 500px frame, corresponds to 25px motion.
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
%                       subpixel depth. (default 2,i.e., 2^-2 = 0.25px).
%                       defaults to 0 (i.e., no subpixel interpolation)
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
%   'params' structure and 'peakPosition'. 
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

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
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
    params.axesHandles(1) = subplot(2,3,[1 4]);
    params.axesHandles(2) = subplot(2,3,2);
    params.axesHandles(3) = subplot(2,3,3);
    params.axesHandles(4) = subplot(2,3,[5 6]);
    for i=1:4
        cla(params.axesHandles(i));
    end
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputVideo, 'strip', params.samplingRate);
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['StripAnalysis() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        RevasMessage('StripAnalysis() is returning results from existing file.',params); 
        
        % try loading existing file contents
        load(outputFilePath,'position', 'timeSec', 'rawPosition', 'peakValueArray','params','peakPosition');
        if nargout > 4
            varargout{1} = params;
        end
        if nargout > 5
            varargout{2} = peakPosition;
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

% get a copy of the reference frame
referenceFrame = params.referenceFrame;

%% Prepare variables before the big loop

[refHeight, refWidth] = size(params.referenceFrame); %#ok<ASGLU>
stripsPerFrame = round(params.samplingRate / params.frameRate);
rowNumbers = round(linspace(1,max([1,height - params.stripHeight + 1]),stripsPerFrame));
params.rowNumbers = rowNumbers;
numberOfStrips = stripsPerFrame * numberOfFrames;

% two columns for horizontal and vertical movements
rawPosition = nan(numberOfStrips, 2);
peakPosition = nan(numberOfStrips, 2);
position = nan(numberOfStrips, 2);

% array for position change
deltaPos = nan(numberOfStrips, 1);
deltaPos(1) = 0;

% array for peak values
peakValueArray = nan(numberOfStrips, 1);

% Populate time array
dtPerScanLine = 1/((height + sum(params.trim)) * params.frameRate);
if length(params.badFrames) > numberOfFrames
    absoluteFrameNo = find(~params.badFrames)-1;
    if ~isrow(absoluteFrameNo)
        absoluteFrameNo = absoluteFrameNo';
    end
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
    numOfAttempts = 2; 
    
    % check if stripheight is larger than searchWindow
    if params.stripHeight >= params.searchWindowHeight
        params.searchWindowHeight = params.stripHeight + 1;
        RevasWarning(['StripAnalysis(): stripHeight (' num2str(params.stripHeight) ') '...
            'is larger than searchWindowHeight (' num2str(params.searchWindowHeight) ').'...
            'Setting searchWindowHeight to ' num2str(params.stripHeight+1)],params);  
    end
else
    numOfAttempts = 1;
end

% needed for reference frame swap operation. if not enabled, offsets are
% zero.
offset = [0 0];

% an array to keep track of last frame where a reference frame was
% swapped.
lastFrameSwap = nan;

% set strip width to frame width, iff params.stripWidth is empty
if isempty(params.stripWidth) || ~IsPositiveInteger(params.stripWidth)
    params.stripWidth = width;
end
stripLeft = max(1,round((width - params.stripWidth)/2));
stripRight = min(width,round((width + params.stripWidth)/2)-1);



%% Extract motion by template matching strips to reference frame

% the big loop. :)

% loop across frames
fr = 1;
while fr <= numberOfFrames

    if ~abortTriggered
        
        % if it's a blink frame, skip it.
        if params.skipFrame(fr)
            % adjust the current time to next frame. (this avoid reading
            % the unused frame)
            if writeResult
                reader.CurrentTime = fr/reader.FrameRate;
            end
            continue;
        else
            % get next frame
            if writeResult
                frame = readFrame(reader);
                if ndims(frame) == 3
                    frame = rgb2gray(frame);
                end
            else
                frame = inputVideo(:,:, fr);
            end
        end
        
        % if GPU is enabled, transfer the frame to GPU memory
        if params.enableGPU
            frame = gpuArray(frame);
        end
        
        % loop across strips within current frame
        for sn = 1:stripsPerFrame
            
            % get current strip
            thisSample = stripsPerFrame * (fr-1) + sn;
            thisStrip = frame(rowNumbers(sn) : (rowNumbers(sn)+params.stripHeight-1),...
                stripLeft:stripRight);
            
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
                        params.rowStart = max(1, round(loc - floor(thisWindowHeight/2)));
                        params.rowEnd = min(refHeight, round(loc + floor(thisWindowHeight/2)));
                        
                        
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
            if params.subpixelDepth ~= 0
                [xPeak, yPeak, peakValue] = ...
                    Interpolation2D(correlationMap, xPeak, yPeak, ...
                        params.neighborhoodSize, params.subpixelDepth, [], params.enableGPU);
            end
            
            % update the traces/stats
            peakValueArray(thisSample) = peakValue;
            peakPosition(thisSample,:) = [xPeak, yPeak - params.stripHeight + params.rowStart - 1];
            rawPosition(thisSample,:) = [xPeak-params.stripWidth-stripLeft+1 ...
                (yPeak - params.stripHeight - rowNumbers(sn) + params.rowStart)] + offset;
            
            % keep a record of amount of motion between successive strips.
            if thisSample > 1
                tempDeltaPos = (diff(rawPosition(thisSample-1:thisSample))/height) ./ ...
                    (diff(timeSec(thisSample-1:thisSample)) / (dtPerScanLine * height / stripsPerFrame));
                deltaPos(thisSample) = sqrt(sum(tempDeltaPos.^2,2)) * stripsPerFrame;
            end
            
            % if adaptive search is enabled, circularly shift the buffer to
            % update the last N positions.
            if params.adaptiveSearch 
                movementHistory = circshift(movementHistory,1,1);
                if thisSample > 1 && ...
                        all(peakValueArray((thisSample-1):thisSample) >= params.minPeakThreshold) && ...
                        all(deltaPos((thisSample-1):thisSample) <= params.maxMotionThreshold)
                    movementHistory(1) = diff(rawPosition(thisSample-1:thisSample,2));
                end
            end

        end % end of frame
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if params.enableReferenceFrameUpdate
            lastFrameStrips = thisSample-stripsPerFrame+1 : thisSample;
            
            % when some strips within a frame results in peak values below
            % our criterion level (or all strips above motion threshold) and
            % when there is already a candidate reference frame, swap the
            % reference frame and compute the offset needed to re-reference
            % upcoming strips to the initial reference frame.
            if fr > 2 && ...
                    sum((peakValueArray(lastFrameStrips) < params.minPeakThreshold) | ...
                        (deltaPos(lastFrameStrips) > params.maxMotionThreshold) ) ...
                    > ((params.swapFrameCriterion) * stripsPerFrame) && ...
                    exist('lastGoodFrame','var') && ...
                    lastFrameSwap(end) ~= fr

                % get a strip from the lastGoodFrame based on local
                % contrast
                lineContrast = medfilt1(std(double(lastGoodFrame),[],2),31);
                [~, maxIx] = max(lineContrast);
                anchorSt = max(1,maxIx - round(height/8));
                anchorEn = min(height, maxIx + round(height/8));
                anchorStrip = lastGoodFrame(anchorSt:anchorEn,:);
                
                % create a struct for full-reference crosscorr.
                anchorOp = struct;
                anchorOp.enableGPU = params.enableGPU;
                anchorOp.corrMethod = params.corrMethod;
                anchorOp.adaptiveSearch = false;
                anchorOp.referenceFrame = params.referenceFrame;
                anchorOp.rowStart = 1;
                anchorOp.rowEnd = refHeight;

                % find the anchor strip in current reference frame
                [cm, xPeakAnchor, yPeakAnchor, ~] = LocateStrip(anchorStrip,anchorOp,struct);
                
                % the location of the anchor strip in new reference frame
                xPeakNew = width;
                yPeakNew = anchorSt + size(anchorStrip,1) - 1;
                
                % offset needed to re-reference the strips to the initial
                % reference frame
                thisOffset = [xPeakAnchor yPeakAnchor] - [xPeakNew yPeakNew];

                if any(thisOffset./height > params.maxMotionThreshold)
                    cmFilt = cm - imgaussfilt(cm,5);
                    [xPeakAnchor, yPeakAnchor, ~] = FindPeak(cmFilt, params.enableGPU);
                    thisOffset = [xPeakAnchor yPeakAnchor] - [xPeakNew yPeakNew];
                end
                
                offset = offset + thisOffset;
                
                % update the reference frame
                params.referenceFrame = lastGoodFrame;
                refHeight = size(lastGoodFrame,1);

                % set a flag so that we don't get stuck here, analyzing the
                % same frame over and over again.
                lastFrameSwap = [lastFrameSwap; fr]; %#ok<AGROW>

                % rewind back one frame
                fr = fr - 1;
                if writeResult
                    reader.CurrentTime = (fr)/reader.FrameRate;
                end

                % give a warning to let user know about the reference frame
                % update
                RevasMessage(['StripAnalysis: Reference frame changed at frame ' num2str(fr) '!'],params);
                
            end
        
            % keep a spare candidate reference frame, in case current one
            % fails. A frame can be a candidate reference frame only when
            % a certain proportion of strips within that frame is above
            % minPeakThreshold.
            if fr > 1 && ...
                    sum((peakValueArray(lastFrameStrips) >= params.minPeakThreshold) | ...
                        (deltaPos(lastFrameStrips) <= params.maxMotionThreshold) ) ...
                    > (params.goodFrameCriterion * stripsPerFrame)
                lastGoodFrame = frame;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % plot peak values and raw traces
        if params.enableVerbosity 
            % show peak values
            scatter(params.axesHandles(2),timeSec,peakValueArray,10,'filled'); 
            hold(params.axesHandles(2),'on');
            plot(params.axesHandles(2),timeSec([1 end]),params.minPeakThreshold*ones(1,2),'-','linewidth',2);
            set(params.axesHandles(2),'fontsize',14);
            xlabel(params.axesHandles(2),'time (sec)');
            ylabel(params.axesHandles(2),'peak value');
            ylim(params.axesHandles(2),[0 1]);
            xlim(params.axesHandles(2),[0 timeSec(thisSample)]);
            hold(params.axesHandles(2),'off');
            grid(params.axesHandles(2),'on');
            
            % plot motion criterion
            scatter(params.axesHandles(3),timeSec,100*deltaPos,10,'filled');
            hold(params.axesHandles(3),'on');
            plot(params.axesHandles(3),timeSec([1 end]),params.maxMotionThreshold*ones(1,2)*100,'-','linewidth',2);
            set(params.axesHandles(3),'fontsize',14);
            xlabel(params.axesHandles(3),'time (sec)');
            ylabel(params.axesHandles(3),'motion (%/fr)');
            xlim(params.axesHandles(3),[0 timeSec(thisSample)]);
            ylim(params.axesHandles(3),[0 min(0.5,max(deltaPos)+0.1)]*100);
            hold(params.axesHandles(3),'off');
            grid(params.axesHandles(3),'on');
            legend(params.axesHandles(3),'off')

            % show useful output traces
            usefulIx = peakValueArray >= params.minPeakThreshold & ...
                deltaPos <= params.maxMotionThreshold;
            tempPos = rawPosition(usefulIx,:);
            plot(params.axesHandles(4),timeSec(usefulIx),tempPos,'-o','linewidth',1.5,'markersize',2);
            set(params.axesHandles(4),'fontsize',14);
            xlabel(params.axesHandles(4),'time (sec)');
            ylabel(params.axesHandles(4),'position (px)');
            legend(params.axesHandles(4),{'hor','ver'});
            yMin = prctile(tempPos,2.5,'all')-20;
            yMax = prctile(tempPos,97.5,'all')+20;
            ylim(params.axesHandles(4),[yMin yMax]);
            xlim(params.axesHandles(4),[0 timeSec(thisSample)]);
            hold(params.axesHandles(4),'off');
            grid(params.axesHandles(4),'on');

            drawnow; % maybe needed in GUI mode.
        end % en of plot
    end % abortTriggered
    
    fr = fr + 1;
end % end of video


% good samples
ix = peakValueArray >= params.minPeakThreshold & ...
    deltaPos <= params.maxMotionThreshold;
position(ix,:) = rawPosition(ix,:);

% restore the initial reference frame 
params.referenceFrame = referenceFrame;

% save the reference frame swap events
params.lastFrameSwap = lastFrameSwap;


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
    scatter(params.axesHandles(1),positionsToBePlotted(:,1), positionsToBePlotted(:,2), ...
        10,  parula(length(timeSec)), 'o' ,'filled','MarkerFaceAlpha',0.2);
    hold(params.axesHandles(1),'off');
    axis(params.axesHandles(1),'image')
    xlim(params.axesHandles(1),[1 size(params.referenceFrame,2)])
    ylim(params.axesHandles(1),[1 size(params.referenceFrame,1)])

    % plot useful position traces.
    plot(params.axesHandles(4),timeSec,position,'-o','linewidth',1.5,'markersize',2);
    set(params.axesHandles(4),'fontsize',14);
    xlabel(params.axesHandles(4),'time (sec)');
    ylabel(params.axesHandles(4),'position (px)');
    legend(params.axesHandles(4),{'hor','ver'});
    yMin = prctile(tempPos,2.5,'all')-20;
    yMax = prctile(tempPos,97.5,'all')+20;
    ylim(params.axesHandles(4),[yMin yMax]);
    xlim(params.axesHandles(4),[0 max(timeSec)]);
    hold(params.axesHandles(4),'off');
    grid(params.axesHandles(4),'on');
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
    save(outputFilePath, 'position', 'rawPosition', 'timeSec', 'params','peakValueArray','peakPosition');
end


%% return the params structure if requested

if nargout > 4
    varargout{1} = params;
end
if nargout > 5
    varargout{2} = peakPosition;
end

