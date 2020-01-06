function [filteredEyePositions, timeSec, varargout ]= FilterEyePosition(inputArgument, params)
%FILTER EYE POSITION fixes temporal gaps in the eye position traces due to
%   blinks or bad strips, applies a series of filters, and then removes
%   interpolated regions. Fixing gaps is required for filter functions to work
%   properly. Interpolated regions are removed after filtering since they do
%   not represent real data. However, if the gaps are smaller than a set
%   threshold duration, interpolation is kept.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputArgument| is a file path for the eye position data that has to
%   have two arrays, |positionDeg| and |timeSec|. Or, if it is not 
%   a file path, then it must be a nxm double array, where m>=2 and n is
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
%   overwrite           : set to true to overwrite existing files.
%                         Set to false to abort the function call if the
%                         files already exist. (default false)
%   enableVerbosity     : set to true to report back plots during execution.
%                         (default false)
%   maxGapDurationMs    : maximum allowable gap duration
%                         in msec. Gaps shorter than this value
%                         will be interpolated in the final
%                         traces.
%   maxPosition         : position threshold in degrees. Any abs(position)
%                         above this value will be removed prior to 
%                         filtering.
%   maxVelocity         : velocity threshold in degrees/second. Any
%                         abs(velocity) above this will be removed prior to
%                         filtering.
%   beforeAfterMs       : buffer zone in miliseconds. When removing
%                         artifacts, n samples corresponding to
%                         beforeAfterMs will also be removed.
%   filters             : an nx1 or 1xn cell array of char arrays/strings for
%                         different types of filters. Any arbitrary
%                         function can be used. Filtering will be applied
%                         in the order indicated in this array.
%                         As of 12/29/31, the following filters are
%                         supported: 'medfilt1', 'notch', 'sgolayfilt'.
%   filterParams        : an nx1 or 1xn cell array of parameters for
%                         corresponding filters in "filters". Each
%                         row of the cell array can contain an array of
%                         parameters.
%   samplingRate        : Sampling rate in Hz for output. By default it's
%                         empty, i.e., input position is filtered without
%                         changing sampling rate. If set to any positive
%                         integer, filtering is done after resampling to
%                         the desired sampling rate.
%   axesHandles         : axes handle for giving feedback. if not
%                         provided or empty, new figures are created.
%                         (relevant only when enableVerbosity is true)
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |filteredEyePositions| is the filtered eye position traces. Rows
%   represent time in ascending order and columns represent different eye
%   traces (horizontal, vertical, or multiple traces from different videos.
%
%   |timeSec|. time in seconds after required interpolation.
%
%   |varargout| is for returning params.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputPath = 'MyFile.mat';
%       params.filters = {@medfilt1, @sgolayfilt};
%       params.filterParams = {11,[3 15]}; 
%                     % 11 is for medfilt1, [3 15] 
%                     % is for sgolayfilt (3 is degree of poly, 15 size of kernel)
%       FilterEyePosition(inputPath, params);
% 
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       params = struct;
%       inputArray = [eyePositionDeg time];
%       filteredPositions = FilterEyePosition(inputArray, params);

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end


%% Determine inputType type.
if ischar(inputArgument)
    % A path was passed in.
    % Read and once finished with this module, write the result.
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

% just one more additional check
% it is to make sure each filter has corresponding parameters
if length(params.filters) ~= length(params.filterParams)
    error('FilterEyePosition: number of filters and number of entries in filterParams do not match.');
end


%% Handle verbosity 

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Filter Eye Position','units','normalized','outerposition',[0.16 0.053 0.67 0.51]);
    if params.enableVerbosity == 1
        params.axesHandles(1) = subplot(1,1,1); % hor and ver
    else
        params.axesHandles(1) = subplot(1,2,1); % hor 
        params.axesHandles(2) = subplot(1,2,2); % ver
    end
       
    for i=1:length(params.axesHandles)
        cla(params.axesHandles(i));
    end
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputArgument, 'filtered');
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['FilterEyePosition() did not execute because it would overwrite existing file. (' outputFilePath ')'], params);
        RevasMessage('FilterEyePosition() is returning results from existing file.',params); 
        
        % try loading existing file contents
        load(outputFilePath,'filteredEyePositions','timeSec');
        if nargout > 2
            varargout{1} = params;
        end
        
        return;
    else
        RevasMessage(['FilterEyePosition() is proceeding and overwriting an existing file. (' outputFilePath ')'], params);  
    end
end


%% Handle inputArgument scenarios

if writeResult
    % check if input file exists
    if ~exist(inputArgument,'file')
        error('FilterEyePosition: eye position file does not exist!');
    end
    
    % load the data
    load(inputArgument,'positionDeg','timeSec');
    eyePositionTraces = positionDeg;

else % inputArgument is not a file path, but carries the eye position data.    
    eyePositionTraces = inputArgument(:,1:size(inputArgument,2)-1);
    timeSec = inputArgument(:,size(inputArgument,2)); % last column is always 'time'
end

% if the dimensions of eyePositionTraces are not appropriate, throw error.
if size(eyePositionTraces,2) >= size(eyePositionTraces,1)
    error('FilterEyePosition: more columns than rows in eye position traces. Make sure position array is mxn where rows represent samples over time');
end


% %% Handle blink/bad frame gaps
% 
% % look for gaps in timeSec
% deltaT = diff(timeSec);
% dt = mean(deltaT);
% gapIndices = deltaT > 2*dt;
% 
% % put a single nan where there is a gap. During interpolation, 
% [timeSec, ix] = sort([timeSec; timeSec(gapIndices)+dt]);
% eyePositionTraces = [eyePositionTraces; nan(sum(gapIndices),size(eyePositionTraces,2))];
% eyePositionTraces = eyePositionTraces(ix,:);


%% Eliminate "lone wolves"
% remove samples and sample pairs that are surrounded by more than two nans
% on each side
surroundedSamples = (conv2(isnan(eyePositionTraces),[1 0 1]','same') == 2 & ...
                    conv2(isnan(eyePositionTraces),[1 1 1]','same') == 2) | ...
                    (conv2(isnan(eyePositionTraces),[1 0 0 1]','same') == 2 & ...
                    conv2(isnan(eyePositionTraces),[1 1 1 1]','same') == 2);
eyePositionTraces(surroundedSamples) = nan;

% compute sampling rate
deltaT = diff(timeSec);
samplingRate = mean(1./deltaT);


%% Remove the artifacts 
% Artifacts are defined as grossly off position
% and/or velocity

velocity = [zeros(1,size(eyePositionTraces,2)); diff(eyePositionTraces,1,1) ./ deltaT];
artifactIndices = abs(eyePositionTraces) >= params.maxPosition | ...
    prod(abs(velocity) >= params.maxVelocity,2) ~= 0;
              
% the following is for removing n samples before and after when
% position exceeds the threshold
nSamples = ceil(params.beforeAfterMs * samplingRate / 1000); 
artifactIndices = conv2(artifactIndices,true(2 * nSamples - 1,1),'same') ~= 0;

% now remove absolutely unuseful parts
eyePositionTraces(artifactIndices) = NaN;


%% Prepare data for filtering.
% nan values will prevent most filters from working properly. So we
% interpolate/extrapolate over them before filtering. After filtering, we
% keep interpolated regions which are shorter than 'params.maxGapDurationMs'.
nanIndices = isnan(sum(eyePositionTraces,2));

% interp1 works separately on each column of eyePositionTraces.
eyePositionInterp = interp1(timeSec(~nanIndices),eyePositionTraces(~nanIndices,:),timeSec,'linear');

% NaN positions have been marked and interpolated between values, but not
% interpolated at leading and trailing position. However we can't filter
% with those NaN value if they exist. So basically we replace them with the 
% next/last scalar available. 
% After filtering, these areas will be replaced with nanIndices and stitched.
tempNans = sum(isnan(eyePositionInterp),2);
firstNonnan = find(~tempNans,1,'first');
lastNonnan = find(~tempNans,1,'last');

eyePositionInterp(1:firstNonnan,:) = eyePositionInterp(firstNonnan+1,:);
eyePositionInterp(lastNonnan:end,:) = repmat(eyePositionInterp(lastNonnan-1,:),size(eyePositionInterp,1)-lastNonnan+1,1);



%% Construct desired filters, in desired order.
% Create any additional filter here, or create and external function which
% only takes in position and some filter parameters. If any other variable
% (such as timeSec or samplingRate) is needed, function must be created
% here to avoid using global variables.

% create a notch filter
function fPos = notch(x, leftCutoff, rightCutoff, filterOrder) %#ok<DEFNU>

    filterObj = designfilt('bandstopiir','FilterOrder',filterOrder, ...
        'HalfPowerFrequency1',leftCutoff,'HalfPowerFrequency2',rightCutoff, ...
        'DesignMethod','butter','SampleRate',samplingRate);
    fPos = filtfilt(filterObj, x);

end


%% Do filtering.

% preallocate memory for the filtered position
filteredEyePositions = eyePositionInterp;

% user asks for more detailed feedback, we store the output of each
% filtering step and show at the end
if params.enableVerbosity > 1
    filteringResults = nan([size(filteredEyePositions), length(params.filters)]);
end
    

% go over each filter type in the order they are given.
for i=1:length(params.filters)
    if ~abortTriggered
    
        currentFilter = params.filters{i}; 
        currentFilterParameters = params.filterParams{i};

        % construct an expression for filter parameters
        paramsStr = [];
        for k=1:length(currentFilterParameters)
            paramsStr = [paramsStr ',' num2str(currentFilterParameters(k))]; %#ok<AGROW>
        end

        % apply current filter
        filteredEyePositions = eval([currentFilter '(filteredEyePositions' paramsStr ')']);
        
        % store the results of each filtering operation
        if params.enableVerbosity > 1
            filteringResults(:,:,i) = filteredEyePositions;
        end
    
    end
end


%% Remove the interpolated regions.
% the following interpolates/stitches gap region appropriately. If the gap
% is tiny, e.g., a few samples long, then we interpolate.. If longer than
% that, we stitch different parts.

maxNumberOfSamples = round(params.maxGapDurationMs * samplingRate / 1000); % samples

% find the interpolated regions
diffInd = diff([0; nanIndices(:); 0]);
start = find(diffInd == 1);
stop = find(diffInd == -1);
dur = stop-start;

% now stitch (i.e., keep interpolation)
regionsToBeFixed = find(dur <= maxNumberOfSamples);
for i=1:length(regionsToBeFixed)
    nanIndices(start(regionsToBeFixed(i)):stop(regionsToBeFixed(i))) = false;
end

filteredFullArray = filteredEyePositions;
filteredEyePositions(nanIndices,:) = nan;



%% visualize results if user requested
if ~abortTriggered && params.enableVerbosity 
    
    if length(params.axesHandles) > 1
        for i=1:length(params.axesHandles)
            hold(params.axesHandles(i),'on');
            plot(params.axesHandles(i), timeSec, eyePositionTraces(:,i), '. ');
        end
    else
        hold(params.axesHandles(1),'on');
        plot(params.axesHandles(1), timeSec, eyePositionTraces, '. ');
    end
    
    % plot all levels of filtering
    if params.enableVerbosity > 1
        forLegend = {'raw'};
        for i=1:length(params.filters)
            tempResult = filteringResults(:,:,i);
            tempResult(nanIndices,:) = nan;
            plot(params.axesHandles(1), timeSec, tempResult(:,1), '-','linewidth',1 + i*.5);
            plot(params.axesHandles(2), timeSec, tempResult(:,2), '-','linewidth',1 + i*.5);
            forLegend{i+1} = params.filters{i};
        end
        legend(params.axesHandles(1),forLegend,'location','best');
        title(params.axesHandles(1),'horizontal');
        title(params.axesHandles(2),'vertical');
        
    else % plot only the overall output
        plot(params.axesHandles(1), timeSec, filteredEyePositions, '-','linewidth',2,'markersize',2);
        legend(params.axesHandles(1),{'raw-hor','raw-ver','filtered-hor','filtered-ver'},'location','best');
    end
    
    % beautify the plot
    for i=1:length(params.axesHandles)
        set(params.axesHandles(i),'fontsize',14);
        xlabel(params.axesHandles(i),'time (sec)');
        ylabel(params.axesHandles(i),'position (deg)');
        ylim(params.axesHandles(i),[nanmin(filteredEyePositions(:)) nanmax(filteredEyePositions(:))] * 1.2);
        xlim(params.axesHandles(i),[0 max(timeSec)]);
        hold(params.axesHandles(i),'off');
        grid(params.axesHandles(i),'on');
    end

end


%% Save filtered data.
if writeResult && ~abortTriggered
    
    try
        % remove pointers to graphics objects
        params = rmfield(params,'commandWindowHandle');
        params = rmfield(params,'axesHandles');
    catch
    end
    
    data.filteredEyePositions = filteredEyePositions;
    data.filteredFullArray = filteredFullArray;
    data.nanIndices = nanIndices;
    data.timeSec = timeSec;
    data.params = params;
    save(outputFilePath,'-struct','data');
end


%% return the params structure if requested
if nargout > 2
    varargout{1} = params;
end


end % keep this 'end' since we have a subfunction above




