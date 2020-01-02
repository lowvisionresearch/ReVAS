function [refFrame, varargout] = MakeReference(inputVideo, params)
%[refFrame, varargout] = MakeReference(inputVideo, params)
%
%   Make a reference frame. Uses the output of 'StripAnalysis.m' to get
%   position shifts for building the reference frame. Uses some heuristics
%   to select good quality strips to build the reference. Most important
%   heuristics are crosscorrelation peak and position shift from previous
%   strip. In other words, a given strip is used in reference frame
%   construction only when it results in a high crosscorrelation peak value
%   and there is not much eye motion since last strip.
%
%   MakeReference takes all input parameters that StripAnalysis does, and
%   some more. 
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
%   enableVerbosity   : set to true to report back plots during execution.(
%                       default false)
%   subpixelForRef    : in octaves, the scaling of the desired level of 
%                       subpixel depth. (default 2,i.e., 2^-2 = 0.25px).
%                       Even if StripAnalysis was run without subpixel
%                       option enabled, this setting is still relevant if
%                       stripHeight (below) for MakeReference is smaller
%                       than the one used for StripAnalysis.
%   rowNumbers        : Row numbers for all strips within one video frame.
%   oldStripHeight    : strip height to be used for strip analysis in 
%                       pixels. (default [], i.e., fail -- must come from
%                       StripAnalysis)
%   newStripHeight    : strip height to be used for making the reference.
%   newStripWidth     : strip width for making the reference.
%   positions         : position traces after compensating for strip
%                       location within parent frame. Also compensates for
%                       trim operation, if any. 
%   timeSec           : time array for positions. in seconds.
%   peakValues        : Peak values. See help StripAnalysis.m for more
%                       info.
%   badFrames         : vector containing the frame numbers of the blink 
%                       frames. (default [])
%   axesHandles       : axes handle for giving feedback. if not provided, 
%                       new figures are created. (relevant only when
%                       enableVerbosity is true)
%   minPeakThreshold  : Minimum peak value threshold (0-1). See help
%                       StripAnalysis.m for more info.
%   maxMotionThreshold: Maximum motion between successive strips.
%                       Specificed in terms of ratio with respect to the
%                       frame height. Value must be between 0 and 1. E.g.,
%                       0.05 for a 500px frame, corresponds to 25px motion.
%   trim              : 1x2 array. number of pixels removed from top and 
%                       bottom of the frame if video was processed in
%                       TrimVideo. (default [0 0] -- [top, bottom]).
%   enhanceStrips     : set to true to do contrast enhancement to
%                       individual strips before summation. (defaults to
%                       true). We use imadjust function for a simple
%                       contrast enhancement.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |refFrame| is a 2D array representing the reference frame.
%
%   |varargout| is a variable output argument holder.
%   varargout{1} = refFrameFilePath
%   varargout{2} = params
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


%% Handle verbosity 

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Make Reference','units','normalized','outerposition',[0.16 0.053 0.67 0.51]);
    params.axesHandles(1) = subplot(2,3,[1 2 4 5]);
    params.axesHandles(2) = subplot(2,3,3);
    params.axesHandles(3) = subplot(2,3,6);
    for i=1:3
        cla(params.axesHandles(i));
    end
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputVideo, 'ref');
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['MakeReference() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        RevasMessage('MakeReference() is returning results from existing file.',params); 
        
        % try loading existing file contents
        load(outputFilePath,'refFrame','params');
        if nargout > 1
            varargout{1} = outputFilePath;
        end
        if nargout > 2
            varargout{2} = params;
        end
        
        return;
    else
        RevasMessage(['MakeReference() is proceeding and overwriting an existing file. (' outputFilePath ')'], params);  
    end
else
    outputFilePath = [];
end


%% Create a reader object if needed and get some info on video

if writeResult
    reader = VideoReader(inputVideo);
    width = reader.Width; 
    height = reader.Height;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    [height, width, numberOfFrames] = size(inputVideo); 
end


%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);


%% Prepare variables before the big loop

newRowNumbers = 1:params.newStripHeight:(height-params.newStripHeight+1);
% newRowNumbers = params.rowNumbers;
% params.newStripHeight = diff(newRowNumbers(1:2));
stripsPerFrame = length(newRowNumbers);
newNumberOfStrips = stripsPerFrame * numberOfFrames;
oldNumberOfStrips = length(params.timeSec) / numberOfFrames;
dtPerScanLine = diff(params.timeSec(1:2)) / diff(params.rowNumbers(1:2));

% estimate the extent of motion, i.e., how much motion occurred during a
% video will determine the size of the final reference frame 
deltaPos = [0; sqrt(sum((diff(params.positions,[],1)/height).^2,2))] * oldNumberOfStrips;
usefulSamples = (params.peakValues >= params.minPeakThreshold) & ...
                (deltaPos <= params.maxMotionThreshold);


% interpolate position for new strip height
newTimeSec = dtPerScanLine * ...
    reshape((0:(numberOfFrames-1)) * (sum(params.trim) + height) + newRowNumbers', newNumberOfStrips, 1);
newUsefulSamples = imresize(double(usefulSamples),size(newTimeSec));
newPositions = interp1(params.timeSec(usefulSamples), params.positions(usefulSamples,:), newTimeSec, 'linear');
newPositions(~newUsefulSamples) = nan;

% if subpixel operation is enabled, i.e. subpixelForRef ~= 0, everything
% needs to upsampled/resized/scaled by 2^subpixelForRef.
sizeFactor = 2^params.subpixelForRef;

% create an accumulator and a counter array based on motion.
minPos = min(params.positions(usefulSamples,:),[],1);
maxPos = max(params.positions(usefulSamples,:),[],1);
refWidth = round((maxPos(1) - minPos(1) + width + 1) * sizeFactor);
refHeight = round((maxPos(2) - minPos(2) + height + 1) * sizeFactor);
accumulator = zeros(refHeight, refWidth);
counter = zeros(refHeight, refWidth);

% set strip width to frame width, iff params.stripWidth is empty
if isempty(params.newStripWidth) || ~IsPositiveInteger(params.newStripWidth)
    params.newStripWidth = width;
end
stripLeft = max(1,round((width - params.newStripWidth)/2));
stripRight = min(width,round((width + params.newStripWidth)/2)-1);
stripWidth = stripRight - stripLeft + 1;


%% plot peak values and delta motion in advance
if ~abortTriggered && params.enableVerbosity 
    
    % show peak value criterion
    scatter(params.axesHandles(2),params.timeSec,params.peakValues,10,'filled'); 
    hold(params.axesHandles(2),'on');
    plot(params.axesHandles(2),params.timeSec([1 end]),params.minPeakThreshold*ones(1,2),'-','linewidth',2);
    set(params.axesHandles(2),'fontsize',14);
    xlabel(params.axesHandles(2),'time (sec)');
    ylabel(params.axesHandles(2),'peak value');
    ylim(params.axesHandles(2),[0 1]);
    xlim(params.axesHandles(2),[0 max(params.timeSec)]);
    hold(params.axesHandles(2),'off');
    grid(params.axesHandles(2),'on');

    % plot motion criterion
    scatter(params.axesHandles(3),params.timeSec,100*deltaPos,10,'filled');
    hold(params.axesHandles(3),'on');
    plot(params.axesHandles(3),params.timeSec([1 end]),params.maxMotionThreshold*ones(1,2)*100,'-','linewidth',2);
    set(params.axesHandles(3),'fontsize',14);
    xlabel(params.axesHandles(3),'time (sec)');
    ylabel(params.axesHandles(3),'motion (%/fr)');
    xlim(params.axesHandles(3),[0 max(params.timeSec)]);
    ylim(params.axesHandles(3),[0 min(0.5,max(deltaPos)+.1)]*100);
    hold(params.axesHandles(3),'off');
    grid(params.axesHandles(3),'on');
    legend(params.axesHandles(3),'off')
    
end

%% Make reference frame

% the big loop. :)

% loop across frames
for fr = 1:numberOfFrames
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
        
        
        % loop across strips within current frame
        for sn = 1:stripsPerFrame
            
            % get current strip
            thisSample = stripsPerFrame * (fr-1) + sn;
            
            % if it's not a useful sample, skip it.
            if any(isnan(newPositions(thisSample,:)))
                continue;
            end
            
            % get new strip
            thisStrip = imresize((frame(newRowNumbers(sn) : min([height,(newRowNumbers(sn)+params.newStripHeight-1)]),...
                stripLeft:stripRight)),sizeFactor);
            
            % enhance contrast
            if params.enhanceStrips
                thisStrip = double(imadjust(thisStrip));
            else
                thisStrip = double(thisStrip);
            end
            
            % compute location in ref frame
            xy = round(sizeFactor * (newPositions(thisSample,:) - minPos + [0 newRowNumbers(sn)]) + 1);
            indY = xy(2):(xy(2)+ size(thisStrip,1) - 1);
            indX = xy(1):(xy(1)+ sizeFactor * stripWidth - 1);
            
            % accumulate
            accumulator(indY, indX) = accumulator(indY, indX) + thisStrip;

            
            % count
            counter(indY, indX) = counter(indY, indX) + 1;
            
        end
        
        
        %% visualize the reference after every frame
        if params.enableVerbosity > 1
            
            % plot reference frame
            axes(params.axesHandles(1)); %#ok<LAXES>
            imagesc(accumulator ./ counter);
            colormap(params.axesHandles(1),gray(256));
            hold(params.axesHandles(1),'on');
            hold(params.axesHandles(1),'off');
            axis(params.axesHandles(1),'image')
            title(params.axesHandles(1),['Frame no: ' num2str(fr)]);
            drawnow;
        end
        
    end

end

%% resize if needed
if sizeFactor > 1
    accumulator = imresize(accumulator, 1/sizeFactor);
    counter = imresize(counter, 1/sizeFactor);
end


%% crop out zero column and rows
zeroRows = sum(counter,2) == 0;
st = find(zeroRows == 0,1,'first');
en = find(zeroRows == 0,1,'last');
ind = [1:st en:length(zeroRows)];
counter(ind,:) = []; 
accumulator(ind,:) = [];

zeroColumns = sum(counter,1) == 0;
st = find(zeroColumns == 0,1,'first');
en = find(zeroColumns == 0,1,'last');
ind = [1:st en:length(zeroColumns)];
counter(:,ind) = []; 
accumulator(:,ind) = [];


%% normalize to get reference Frame
refFrameZero = accumulator ./ counter;

% convert to uint8
refFrame = uint8(refFrameZero);

%% replace zero-padded regions with noise
zeroInd = counter == 0;
noise = datasample(refFrame(~zeroInd),sum(zeroInd(:)),'replace',false);
refFrame(zeroInd) = noise;


%% visualize reference frame
if ~abortTriggered && params.enableVerbosity 
    % plot reference frame
    axes(params.axesHandles(1));
    imagesc(refFrame);
    colormap(params.axesHandles(1),gray(256));
    hold(params.axesHandles(1),'on');
    hold(params.axesHandles(1),'off');
    axis(params.axesHandles(1),'image')
    xlim(params.axesHandles(1),[1 size(refFrame,2)])
    ylim(params.axesHandles(1),[1 size(refFrame,1)])
    title(params.axesHandles(1),'Reference Frame')

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
    save(outputFilePath, 'refFrame','refFrameZero','params');

end


%% return the params structure if requested

if nargout > 1
    varargout{1} = outputFilePath;
end
if nargout > 2
    varargout{2} = params;
end










