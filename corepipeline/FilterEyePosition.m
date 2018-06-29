function [filteredEyePosition, outputFilePath, parametersStructure] = ...
    FilterEyePosition(inputArgument, parametersStructure)
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
%   have two arrays, |eyePositionTraces| and |timeArray|. Or, if it is not 
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
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
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
%   filterTypes         : an nx1 cell array of function pointers for
%                         different types of filters. Any arbitrary
%                         function can be used. Filtering will be applied
%                         in the order indicated in this array.
%   filterParameters    : an nx1 cell array of parameters for
%                         corresponding filters in "filterTypes". Each
%                         row of the cell array can contain an array of
%                         parameters.
%   axesHandles         : axes handle for giving feedback. if not
%                         provided or empty, new figures are created.
%                         (relevant only when enableVerbosity is true)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputPath = 'MyFile.mat';
%       parameterStructure.overwrite = 0;
%       parameterStructure.maxGapDurationMs = 10; %ms
%       parameterStructure.filterTypes = {@medfilt1, @sgolayfilt};
%       parameterStructure.filterParameters = {11,[3 15]}; 
%                     % 11 is for medfilt1, [3 15] 
%                     % is for sgolayfilt (3 is degree of poly, 15 size of kernel)
%       FilterEyePosition(inputPath, parameterStructure);
% 
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputArray = [eyePosition time];
%       filteredPosition = FilterEyePosition(inputArray, parameterStructure);

%% Handle misusage 
if nargin<1
    error('FilterEyePosition needs at least two input arguments.')
end

if nargin<2
    error('parametersStructure is not provided, filtering cannot proceed.');
end 

if ~isstruct(parametersStructure)
    error('''parametersStructure'' must be a struct.');
end

%% Set parameters to defaults if not specified.
if ~isfield(parametersStructure,'overwrite')
    overwrite = false;
else
    overwrite = parametersStructure.overwrite;
end

if ~isfield(parametersStructure,'maxGapDurationMs')
    maxGapDurationMs = 10;
    RevasWarning('using default parameter for maxGapDurationMs', parametersStructure);
else
    maxGapDurationMs = parametersStructure.maxGapDurationMs;
end

if ~isfield(parametersStructure,'filterTypes')
    filterTypes = {@sgolayfilt};
    RevasWarning('using default parameter for filterTypes', parametersStructure);
else
    filterTypes = parametersStructure.filterTypes;
end

if ~isfield(parametersStructure,'filterParameters')
    filterParameters = {[3 15]};
    RevasWarning('using default parameter for filterParameters', parametersStructure);
else
    filterParameters = parametersStructure.filterParameters;
end

if ~isfield(parametersStructure,'enableVerbosity')
    verbosity = false;
else
    verbosity = parametersStructure.enableVerbosity;
end

if ~isfield(parametersStructure,'axesHandles')
    axesHandles = [];
else
    axesHandles = parametersStructure.axesHandles;
end

%% Handle |inputArgument| scenarios.
if ischar(inputArgument) % inputArgument is a file path
    outputFilePath = [inputArgument(1:end-4) '_filtered.mat'];
    
    % Handle overwrite scenarios
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~overwrite && exist(outputFilePath, 'file')
        RevasWarning(['FilterEyePosition() did not execute because it would overwrite existing file. (' outputFilePath ')'], parametersStructure);
        return;
    else
        RevasWarning(['FilterEyePosition() is proceeding and overwriting an existing file. (' outputFilePath ')'], parametersStructure);  
    end
    
    % check if input file exists
    if ~exist(inputArgument,'file')
        error('eye position file does not exist!');
    end
    
    % load the data
    data = load(inputArgument);
    eyePositionTraces = data.eyePositionTraces;
    timeArray = data.timeArray;

else % inputArgument is not a file path, but carries the eye position data.
    outputFilePath = [];
    
    eyePositionTraces = inputArgument(:,1:size(inputArgument,2)-1);
    timeArray = inputArgument(:,size(inputArgument,2)); % last column is always 'time'
    
end

% if the dimensions of eyePositionTraces are not appropriate, throw error.
if size(eyePositionTraces,2) >= size(eyePositionTraces,1)
    error('more columns than rows in eye position traces.');
end


%% Prepare data for filtering.
% nan values will prevent most filters from working properly. So we
% interpolate/extrapolate over them before filtering. After filtering, we
% keep interpolated regions which are shorter than 'maxGapDurationMs'.
nanIndices = isnan(sum(eyePositionTraces,2));

for i=1:size(eyePositionTraces,2)
    eyePositionTraces(:,i) = interp1(timeArray(~nanIndices),eyePositionTraces(~nanIndices,i),timeArray,'linear');
end

% Notch filter for 30Hz and 60 Hz removal - addition by JG
% Butterworth IIR 2nd order filter
% we can't select the samplingRate in the same maner as in StripAnalysis:
% as it is not accessible in parametersStructure from the postproc window
% TO DO : add samplingRate as a global variable

% NaN positions have been marked and interpolated between values, but not
% interpolated at leading and trailing position. However we can't filter
% with those NaN value if they exist. So basically we replace them with the 
% next/last scalar available. 
% After filtering, these areas will be replaced with nanIndices and stitched.

% Replace leading and trailing NaN value with first and last data
% 'fillmissing' function exist for Matlab version >=2016b
for i=1:size(eyePositionTraces,2)
	eyePositionTraces(1:find(~any(isnan(eyePositionTraces),2),1,'first')-1,i)=eyePositionTraces(find(~any(isnan(eyePositionTraces),2),1,'first'),i);
	eyePositionTraces(find(~any(isnan(eyePositionTraces),2),1,'last')+1:end,i)=eyePositionTraces(find(~any(isnan(eyePositionTraces),2),1,'last'),i);
end
if parametersStructure.FirstPrefilter
    samplingRate = parametersStructure.samplingRate;%540;
    d = designfilt('bandstopiir','FilterOrder',2, ...
                   'HalfPowerFrequency1',29,'HalfPowerFrequency2',31, ...
                   'DesignMethod','butter','SampleRate',samplingRate);
    for i=1:size(eyePositionTraces,2)
        eyePositionTraces(:,i)=filtfilt(d,eyePositionTraces(:,i));
    end
    clear d;
end

if parametersStructure.SecondPrefilter
    d2 = designfilt('bandstopiir','FilterOrder',2, ...
                   'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
                   'DesignMethod','butter','SampleRate',samplingRate);
    for i=1:size(eyePositionTraces,2)
        eyePositionTraces(:,i)=filtfilt(d2,eyePositionTraces(:,i));
    end
    clear d2;
end

%% Do filtering.
% note: although some filters can be applied to 2D arrays directly (e.g., 
% sgolayfilt), to preserve generality, we will filter each position
% column separately.

% preallocate memory for the filtered position
filteredEyePosition = eyePositionTraces;

% go over each filter type in the order they are given.
for i=1:length(filterTypes)
    
    currentFilter = filterTypes{i}; %#ok<NASGU>
    currentFilterParameters = filterParameters{i};
    
    % construct an expression for filter parameters
    parameterStr = [];
    for k=1:length(currentFilterParameters)
        parameterStr = [parameterStr ',' num2str(currentFilterParameters(k))]; %#ok<AGROW>
    end
    
    % go over each eye position column
    for j=1:size(eyePositionTraces,2)
        filteredEyePosition(:,j) = eval(...
            ['currentFilter(filteredEyePosition(:,j)' parameterStr ')']);
    end

end


%% Remove the interpolated regions.
% the following interpolates/stitches gap region appropriately. If the gap
% is tiny, e.g., a few samples long, then we interpolate.. If longer than
% that, we stitch different parts.

maxNumberOfSamples = round(maxGapDurationMs/(1000*diff(timeArray(1:2)))); % samples

% find the interpolated regions
diffInd = diff([0; nanIndices(1:end-1); 0]);
start = find(diffInd == 1);
stop = find(diffInd == -1);
dur = stop-start;

% now stitch
regionsToBeFixed = find(dur < maxNumberOfSamples);
for i=1:length(regionsToBeFixed)
    nanIndices(start(regionsToBeFixed(i)):stop(regionsToBeFixed(i))) = false;
end

%% Remove the artifacts.
% while we are at it, also remove the parts where position is grossly off
md = nanmedian(filteredEyePosition,1);
sd = nanstd(filteredEyePosition,[],1);

d = 5; % times the standard deviation from median will be removed
remove = (filteredEyePosition(:,1) > (md(1)+d*sd(1))) | (filteredEyePosition(:,1) < (md(1)-d*sd(1))) |...
         (filteredEyePosition(:,2) > (md(2)+d*sd(2))) | (filteredEyePosition(:,2) < (md(2)-d*sd(2)));
beforeAfter = round(0.020/diff(timeArray(1:2))); 
remove = conv(double(remove),ones(beforeAfter,1),'same')>0;

% now remove absolutely unuseful parts
filteredEyePosition(nanIndices | remove,:) = NaN;


%% Save filtered data.
if ~isempty(outputFilePath)
    data.eyePositionTraces = filteredEyePosition;
    
    try
        % remove pointers to graphics objects
        parametersStructure = rmfield(parametersStructure,'commandWindowHandle');
        parametersStructure = rmfield(parametersStructure,'axesHandles');
    catch
    end
    
    data.parametersStructure = parametersStructure;
    save(outputFilePath,'-struct','data');
end

%% Give feedback if user requested.
if verbosity
   if ishandle(axesHandles)
       axes(axesHandles(2));
   else
       figure(1453);
       axes(gca);
   end
   cla;
   ax = gca;
   ax.ColorOrderIndex = 1;
   plot(timeArray,eyePositionTraces); hold on;
   ax.ColorOrderIndex = 1;
   plot(timeArray,filteredEyePosition,'LineWidth',2);
   xlabel('Time (sec)');
   ylabel('Eye position');
   legend('Hor','Ver')
end
