function [inputArgument, params, varargout] = ...
    FindSaccadesAndDrifts(inputArgument,  params)
%FIND SACCADES AND DRIFTS saccade and drift parsing.
%
%   Offers multiple algorithms: 
%         1) I-VT algorithm : simple velocity threshold for saccades, and
%            the rest is considered drifts (except for samples during
%            blinks/data loss) 
%         2) Klielg & Engbert 2003 algorithm. Velocity threshold based on
%            vectorial sum of median horizontal and vertical velocities.
%            Usually good for microsaccade detection. May fail with long
%            recordings.
%         3) A hybrid of Nyström & Holmqvitz 2010 and Klielg & Engbert 2003
%            algorithms. Applies EK algorithm in a moving window and after
%            identifying saccade regions, travels back and forth from
%            saccade peak to fine-tune boundaries. Based on our experience,
%            barring the machine learning based ones, this is the best
%            algorithm for sample-wise saccade detection. Therefore, this
%            is the default setting.
%
%    On top of these algorithms, we also use some heuristics to further
%    improve sample-to-sample classification accuracy. For instance, if two
%    saccade events are too close each other in time (e.g., <10ms), most
%    likely they are part of the same event. Likewise, we use minimum and
%    maximum allowable saccade amplitudes and durations to do further
%    cleaning up event labels. If a set of samples are labeled as saccades
%    but fail to satisfy these requirements, they go into blink/data loss
%    category. Not a big deal for AOSLO recordings where eye movements are
%    already small, but it may improve classification a great deal for
%    larger fov recordings.
%
%   Note 1: This function always uses vectorial velocity, i.e., horizontal
%   and vertical components are vector-summed, regardless of the selected
%   algorithm. When only one dimension is provided via 'positionDeg', this
%   function still works.
%
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputArgument| is a file path for the eye position data that has to
%   have two arrays, |positionDeg| and |timeSec|. Or, if it is not 
%   a file path, then it must be a nxn double array, where m>=2 and n is
%   the number of data points. The last column of |inputArgument| is always
%   treated as the time signal. The other columns are treated as eye
%   position signals, which will be subjected to filtering. Typically, 
%   |inputArgument| would be an nx3 array, where the first two columns 
%   represent the horizontal and vertical eye positions, respectively, and
%   the last column has the time array.
%   The result is that the filtered version of this file is stored with
%   '_filtered' appended to the original file name. If |inputArgument| 
%   is not a file name but actual eye position data, then the filtered 
%   position data are not stored, and the second output argument is an empty array.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite               : set to true to overwrite existing files.
%                             Set to false to abort the function call if the
%                             files already exist. (default false)
%   enableVerbosity         : set to true to report back plots during execution.
%                             (default false)
%   algorithm               : Can be 'ivt', 'ek', or 'hybrid', See text
%                             above for explanation of each
%                             algorithm. default 'hybrid'.
%   velocityMethod          : Method to use to calculate velocity. 
%                               1 = using |diff|, 
%                               2 = (x_(n+1) - x_(n-1)) / 2 delta t).
%                 function_handle = a function handle to a custom
%                             velocity computation function. The function
%                             must take only timeSec and positionDeg as two
%                             arguments in that order, and must return
%                             velocity at the same size as the positionDeg.
%                             (default 2)
%   axesHandles             : axes handle for giving feedback. if not
%                             provided or empty, new figures are created.
%                             (relevant only when enableVerbosity is true)
%
%   ----- some heuristics --------
%
%   minInterSaccadeInterval : if two consecutive saccades are closer in time
%                             less than this value (in ms), then they are
%                             stitched back to back. (default 15)
%   minSaccadeAmplitude     : minimum (micro)saccade amplitude in deg (default 0.1)
%   maxSaccadeAmplitude     : maximum (micro)saccade amplitude in deg (default 10)
%   maxSaccadeDuration      : maximum saccade duration in ms. (default 100)
%   minSaccadeDuration      : minimum saccade duration in ms. (default 6)
%
%   ----- for I-VT algorithm  --------
%
%   velocityThreshold       : in deg/sec. (default 25)
%
%   ----- for Engbert & Kliegl algorithm  --------
%
%   lambdaForPeak           : multiplier specifying the median-based velocity
%                             threshold. Used for finding saccades in this
%                             algorithm but it is used only for saccade
%                             peaks if the 'hybrid' option is selected. In
%                             that case, a secondary lambda value is used
%                             for fine-tuning onsets and offsets of saccade
%                             events. (default 6)
%
%   ----- for the Hybrid algorithm (NH2010 and EK2003)  --------
%
%   windowSize              : Moving window size in ms. Velocity
%                             threshold is obtained within this video,
%                             which makes this algorithm adaptive to noise
%                             in data.
%   lambdaForOnsetOffset    : multiplier specifying a secondary velocity
%                             threshold for more accurate detection of onset
%                             and offset of a saccade. (default 3)
%   
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |inputArgument| is the same as input. directly passed from input.
%
%   |params| structure with saccades, drifts, and labels, as fields.
%           |saccades| is an array of a saccade structure, each of which contains detailed
%           info about a single saccade.
%           |drifts| is the same as saccades, except it has info on drift "events". 
%           |labels| is an array of integers representing event labels. 1: saccade,
%           2: drift, 3: blink/data loss.
%
%   varargout{1} = st.
%   varargout{2} = en.
%   varargout{3} = driftSt.
%   varargout{4} = driftEn.


%% Determine inputType type.
if ischar(inputArgument)
    % A path was passed in.
    % Read and once finished with this module, write the result.
    writeResult = true;
else
    % a position + time array was passed.
    % Do not write the result; 
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


%% Handle GUI mode
% params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
else
    logBox = [];
end

% params will have access to a uicontrol object in GUI mode. so if it does
% not already have that, create the field and set it to false so that this
% module can be used without the GUI
if ~isfield(params,'abort')
    params.abort.Value = false;
end

%% Handle verbosity 
% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Find Saccades & Drifts',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.67 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');

    params.axesHandles(1) = subplot(1,1,1); 
end

if params.enableVerbosity
    cla(params.axesHandles(1));
    tb = get(params.axesHandles(1),'toolbar');
    tb.Visible = 'on';
end

%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputArgument, 'sacsdrifts');
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['FindSaccadesAndDrifts() did not execute because it would overwrite existing file. (' outputFilePath ')'], logBox);
        RevasMessage('FindSaccadesAndDrifts() is returning results from existing file.',logBox); 
        
        % try loading existing file contents
        load(outputFilePath,'saccades','drifts','labels','st','en','driftSt','driftEn');
        params.saccades = saccades;
        params.drifts = drifts;
        params.labels = labels;
        if nargout > 3
            varargout{1} = st;
        end
        if nargout > 4
            varargout{2} = en;
        end
        if nargout > 5
            varargout{3} = driftSt;
        end
        if nargout > 6
            varargout{4} = driftEn;
        end
        
        return;
    else
        RevasMessage(['FindSaccadesAndDrifts() is proceeding and overwriting an existing file. (' outputFilePath ')'], logBox);  
    end
end


%% Handle inputArgument scenarios

if writeResult
    % check if input file exists
    if ~exist(inputArgument,'file')
        error('FindSaccadesAndDrifts: eye position file does not exist!');
    end
    
    % load the data
    load(inputArgument,'positionDeg','timeSec');

else % inputArgument is not a file path, but carries the eye position data.    
    positionDeg = inputArgument(:,1:size(inputArgument,2)-1);
    timeSec = inputArgument(:,size(inputArgument,2)); % last column is always 'time'
end

% if the dimensions of eyePositionTraces are not appropriate, throw error.
if size(positionDeg,2) >= size(positionDeg,1)
    error('FindSaccadesAndDrifts: more columns than rows in eye position traces. Make sure position array is mxn where rows represent samples over time');
end

% some additional checks for timeSec and positionDeg
if isempty(timeSec) || isempty(positionDeg) || ~isnumeric(timeSec) || ...
        ~isnumeric(positionDeg) || size(timeSec,2)~=1 || ...
        size(timeSec,1)~=size(positionDeg,1)
    error('FindSaccadesAndDrifts: improper input arguments. Check ''timeSec'' or ''positionDeg''.');
end

% assume uniform temporal sampling. (eye position traces should be filtered
% and artifacts should be removed prior to call to this function).
dt = mode(diff(timeSec,1,1)); 



%% Compute vectorial velocity

if isa(params.velocityMethod, 'function_handle')
    velocity = params.velocityMethod(timeSec, positionDeg);
    
else
    % - Method 1: v_x(t) = [x(t + \Delta t) - x(t)] / [\Delta t]
    if params.velocityMethod == 1
        velocity = [zeros(1,size(positionDeg,2)); 
            diff(positionDeg,1,1) / dt];

     % - Method 2: v_x(t) = [x(t + \Delta t) - x(t - \Delta t)] / [2 \Delta t]
    elseif params.velocityMethod == 2   
        velocity = [zeros(1,size(positionDeg,2)); 
            (positionDeg(3:end,:) - positionDeg(1:end-2,:)) / (2 * dt);
            zeros(1,size(positionDeg,2)) ];
    else
        error('FindSaccadesAndDrifts: unknown velocityMethod!');
    end
end

% compute vectorial velocity
vectorialVelocity = sqrt(sum(velocity.^2,2));

% initialize labels
labels = zeros(size(timeSec));
labels(isnan(vectorialVelocity)) = 3; % blinks / data losses


%% Find saccades.

switch params.algorithm
    case 'ivt'
        velocityThreshold = params.velocityThreshold; 
        
    case 'ek'
        sd = mad(vectorialVelocity,1);
        mu = nanmedian(vectorialVelocity);
        velocityThreshold = mu + sd * params.lambdaForPeak;
        
    case 'hybrid'
        windowSizeSamples = RoundToOdd(params.windowSize/(1000*dt));
        sd = movmad(vectorialVelocity, windowSizeSamples,'omitnan');
        mu = movmedian(vectorialVelocity, windowSizeSamples,'omitnan');
        velocityThreshold = mu + sd * params.lambdaForPeak;
        
    otherwise
        error('FindSaccadesAndDrifts: unknown classification algorithm');
end

% mark indices where velocity is above the threshold
aboveThreshold = velocityThreshold < vectorialVelocity;

if sum(aboveThreshold) == 0
    labels(labels ~= 3) = 2; 
    saccades = [];
    st = [];
    en = [];
else

    % compute saccade onset and offset indices
    [onsets, offsets] = GetEventOnsetsAndOffsets(aboveThreshold);

    % assign timestamps
    st = timeSec(onsets);
    en = timeSec(offsets);

    % if using the method of Nystrom & Holmqvitz, find the peak
    % velocity within each segmented section and travel back and forward to
    % find the onset/offset timestamps. Use a different (lower) velocity
    % threshold for this purpose.
    if contains(params.algorithm, 'hybrid')
        % first compute the secondary threshold
        secondaryThreshold = mu + sd * params.lambdaForOnsetOffset;

        % now revise onset/offsets
        [st, en] = FineTuneOnsetOffset(st,en,timeSec,vectorialVelocity,secondaryThreshold);
    end


    %% Segment into saccade events, given some constraints on min/max durations

    % merge events that are too close to each other in time
    [st, en] = MergeSaccades(st, en, params.minInterSaccadeInterval/1000);

    % Get saccade properties
    saccades = GetEventProperties(positionDeg,timeSec,vectorialVelocity,st,en);

    % remove events with duration shorter than minDuration or larger than
    % maxDuration, also remove really small or large saccades
    durations = en - st;
    amplitudes = [saccades.vectorAmplitude]';
    toRemove = durations < params.minSaccadeDuration/1000 | ...
        durations > params.maxSaccadeDuration/1000 | ...
        amplitudes < params.minSaccadeAmplitude | ...
        amplitudes >= params.maxSaccadeAmplitude;

    % update labels
    labels(IndicesForRange(st(~toRemove),en(~toRemove),timeSec)) = 1;

    % also get the indices which are marked as part of this event type but did
    % not satisfy the constraints.
    labels(IndicesForRange(st(toRemove),en(toRemove),timeSec)) = 2;

    % don't forget to remove the stuff we're supposed to remove.
    st(toRemove) = [];
    en(toRemove) = [];
    saccades(toRemove) = [];

    %% Drifts

    % anything other than blinks/data loss and saccades are considered drifts.
    labels(labels == 0) = 2; % drifts

end


driftIndices = labels == 2;
if sum(driftIndices) > 0
    % compute drift onset and offset indices
    [driftOnsets, driftOffsets] = GetEventOnsetsAndOffsets(driftIndices);

    % assign timestamps
    driftSt = timeSec(driftOnsets);
    driftEn = timeSec(driftOffsets);

    % Get drift properties
    drifts = GetEventProperties(positionDeg,timeSec,vectorialVelocity,driftSt,driftEn);
else
    
    drifts = [];
    driftSt = [];
    driftEn = [];
    labels(driftIndices) = 3;
end

%% Verbosity for Results.
if params.enableVerbosity && ~params.abort.Value
    
    sacIx = 1*(labels == 1);
    sacIx(sacIx==0) = nan;
    blinkIx = 1*(labels == 3);
    blinkIx(blinkIx==0) = nan;
    
    cla(params.axesHandles(1));
    ph = [];
    style = {'-','-'};
    cols = lines(8);
    
    dataLims = [nanmin(positionDeg(:)) nanmax(positionDeg(:))];
    ylims = mean(dataLims) + diff(dataLims) * 1.2 * [-1 1]/2;
    for i=1:2
        hold(params.axesHandles(1),'on');
        ph(i) = plot(params.axesHandles(1), timeSec, positionDeg(:,i),style{i}, 'linewidth',1.5,'color',cols(i,:)); %#ok<AGROW>
        sh(1) = plot(params.axesHandles(1), timeSec, sacIx.*positionDeg(:,i),style{i}, 'linewidth',4,'color',cols(i,:));
    end
    sh(2) = plot(params.axesHandles(1), timeSec, blinkIx.*zeros(size(timeSec)),style{i}, 'linewidth',4,'color','k');
    
    set(params.axesHandles(1),'fontsize',10);
    ylim(params.axesHandles(1), ylims);
    xlim(params.axesHandles(1),[0 max(timeSec)]);
    hold(params.axesHandles(1),'off');
    grid(params.axesHandles(1),'on');
    ylabel(params.axesHandles(1),'position (deg)');
    xlabel(params.axesHandles(1),'time (sec)');
    legend(params.axesHandles(1),[ph sh],{'hor-pos','ver-pos','saccade','blink/data loss'},...
        'orientation','horizontal','location','best');
    
end


%% Save filtered data

% remove unnecessary fields
abort = params.abort.Value;
params = RemoveFields(params,{'logBox','axesHandles','abort'}); 

if writeResult && ~abort
    
    save(outputFilePath,'saccades','drifts','labels','params',...
        'st','en','driftSt','driftEn');
end

% append events under params structure
params.saccades = saccades;
params.drifts = drifts;
params.labels = labels;

%% handle variable output arguments

if nargout > 3
    varargout{1} = st;
end
if nargout > 4
    varargout{2} = en;
end
if nargout > 5
    varargout{3} = driftSt;
end
if nargout > 6
    varargout{4} = driftEn;
end


end % keep this 'end' since we have some functions below



%%
function s = GetEventProperties(pos,t,vel,st,en)

    hor = pos(:,1);
    ver = pos(:,2);

    % preallocate memory
    s = repmat(GetEmptyEventStruct, length(st),1);

    for i=1:length(st)

        % extract saccade parameters
        stIx = find(t == st(i));
        enIx = find(t == en(i));
        s(i).onsetTime = st(i);
        s(i).offsetTime = en(i);
        s(i).onsetIndex = stIx;
        s(i).offsetIndex = enIx;
        s(i).duration = en(i) - st(i);
        s(i).xStart = hor(stIx);
        s(i).xEnd = hor(enIx);
        s(i).yStart = ver(stIx);
        s(i).yEnd = ver(enIx);
        s(i).xAmplitude = s(i).xEnd - s(i).xStart;
        s(i).yAmplitude = s(i).yEnd - s(i).yStart;
        s(i).vectorAmplitude = sqrt(s(i).xAmplitude.^2 + s(i).yAmplitude.^2);
        s(i).direction = atan2d(s(i).yAmplitude, s(i).xAmplitude);
        s(i).peakVelocity = max(vel(stIx:enIx));
        s(i).meanVelocity = nanmean(vel(stIx:enIx));
        s(i).maximumExcursion = max(sqrt((hor(stIx:enIx) - hor(stIx)).^2 + (ver(stIx:enIx) - ver(stIx)).^2));
    end
end

%%
function eventStruct = GetEmptyEventStruct
    eventStruct.duration = [];
    eventStruct.onsetIndex = [];
    eventStruct.offsetIndex = [];
    eventStruct.onsetTime = [];
    eventStruct.offsetTime = [];
    eventStruct.xStart = [];
    eventStruct.xEnd = [];
    eventStruct.yStart = [];
    eventStruct.yEnd = [];
    eventStruct.xAmplitude = [];
    eventStruct.yAmplitude = [];
    eventStruct.vectorAmplitude = [];
    eventStruct.direction = [];
    eventStruct.peakVelocity = [];
    eventStruct.meanVelocity = [];
    eventStruct.maximumExcursion = [];
end

%%
function [onsets, offsets] = GetEventOnsetsAndOffsets(indices)

    onsetOffset = [0; diff(indices)];
    onsets = find(onsetOffset == 1);
    offsets = find(onsetOffset == -1);
    
    if isempty(onsets) || isempty(offsets)
        return;
    end

    % address missing onset at the beginning
    if offsets(1) < onsets(1)
        onsets = [1; onsets];
    end

    % address missing offset at the end
    if offsets(end) < onsets(end)
        offsets = [offsets; onsets(end)];
    end
end


%% Round to an odd number
function n = RoundToOdd(n)

    n = round(n);
    ind = mod(n,2) == 0;
    n(ind) = n(ind)+1;

end

%% Fine-tunes onset/offset labels using a secondary threshold
function [st, en] = FineTuneOnsetOffset(st,en,t,vel,threshold)

    if length(st) ~= length(en)
        error('FindSaccadesAndDrifts: onset and offset arrays have different lengths');
    end

    dt = diff(t(1:2)); % assumes regular sampling
    searchWindow = round(0.2 / dt); % in samples

    for i=1:length(st)
        
        % get the indices for this particular saccade
        thisSaccade = IndicesForRange(st(i),en(i),t);

        % find where the peak velocity occured in this saccade 
        [~, ix] = max(vel(thisSaccade));

        % compute the index of peak velocity in the original signal
        peakVelIndex = thisSaccade(1) + ix - 1;

        % ONSET
        % now go back in time to find the onset. BUT don't go too much ;)
        startIndexForSearch = max([1,(peakVelIndex - searchWindow + 1)]);
        acceleration = [0; diff(vel(startIndexForSearch : peakVelIndex))/dt];
        newOnsetIndex = find((acceleration < 0) & ...
            vel(startIndexForSearch : peakVelIndex) < ...
            threshold(startIndexForSearch : peakVelIndex), 1, 'last') - 1; 

        % search might end up with an empty index if velocity never goes below
        % threshold. in that case, use the original timestamp
        if ~isempty(newOnsetIndex)
            % convert that to timestamp
            tempSt = t(startIndexForSearch + newOnsetIndex - 1);

            % check if it's within a reasonable temporal distance from peak
            if abs(tempSt - t(peakVelIndex)) < 0.03
                st(i) = tempSt;
            end
        else
            st(i) = nan;
            continue;
        end

        % OFSET
        endIndexForSearch = min([length(t),(peakVelIndex + searchWindow)]);
        newOffsetIndex = find(vel(peakVelIndex : endIndexForSearch) < ...
             threshold(peakVelIndex : endIndexForSearch),1,'first');

        % search might end up with an empty index if velocity never goes below
        % threshold. in that case, use the original timestamp
        if ~isempty(newOffsetIndex)
            % convert that to timestamp
            tempEn = t(peakVelIndex + newOffsetIndex - 1);

            % check if it's within a reasonable temporal distance from peak
            if abs(tempSt - t(peakVelIndex)) < 0.12
                en(i) = tempEn;
            end
        else
            en(i) = nan;
        end
    end

    % remove problematic ones (those that are only partially recorded)
    ix = isnan(st) | isnan(en);
    st(ix) = [];
    en(ix) = [];

end

%% A helper function to get indices for the given list of start and end timestamps
function ind = IndicesForRange(st,en,t)

    ind = [];
    for i=1:length(st)
        ind = [ind; (find(t==st(i)):find(t==en(i)))']; %#ok<AGROW>
    end

end


%% Merge saccades that are too close in time
function [st, en] = MergeSaccades(st, en, minInterEventInterval)

    % loop until there is no pair of consecutive events closer than
    % *minInterEventInterval*
    while true
        interEventInterval = (st(2:end)-en(1:end-1));
        toBeMerged = find(interEventInterval < minInterEventInterval);
        if isempty(toBeMerged)
            break;
        end

        st(toBeMerged+1) = [];
        en(toBeMerged) = [];

    end

end

