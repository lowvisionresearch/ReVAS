function [filteredEyePosition, outputFilePath, parametersStructure] = ...
    FilterEyePosition(inputArgument, parametersStructure)
%FILTER EYE POSITION fixes temporal gaps in the eye position traces due to
%   blinks or bad strips, applies a series of filters, and then removes
%   interpolated regions. Fixing gaps is required for filter functions to work
%   properly. Interpolated regions are removed after filtering since they do
%   not represent real data. However, if the gaps are smaller than a set
%   threshold duration, interpolation is kept.
%
%   if 'inputArgument' is a file path for the eye position data, then it
%   has to have two arrays, 'eyePositionTraces' and 'timeArray'. 
%
%   if 'inputArgument' is not a file path then it must be a nxm double
%   array, where m>=2 and n is the number of data points. The last column of 
%   'inputArgument' is always treated as the time signal. The other columns
%   are treated as eye position signals, which will be subjected to
%   filtering. Typically, 'inputArgument' would be an nx3 array, where the
%   first two columns represent the horizontal and vertical eye positions,
%   respectively, and the last column has the time array.
%  
%   The output of this process is stored with "_filtered" appended to the
%   input file name. If 'inputArgument' is not a file name but actual
%   eye position data, then the filtered position data are not stored, and
%   the second output argument is an empty array.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite           :   determines whether an existing output
%                           file should be overwritten and replaced if 
%                           it already exists. if 'inputArgument'
%                           is not a file name, overwrite is ignored.
%   maxGapDurationMs    :   maximum allowable gap duration
%                           in msec. Gaps shorter than this value
%                           will be interpolated in the final
%                           traces.
%   filterTypes         :   an nx1 cell array of function pointers for
%                           different types of filters. Any arbitrary
%                           function can be used. Filtering will be applied
%                           in the order indicated in this array.
%   filterParameters    :   an nx1 cell array of parameters for
%                           corresponding filters in "filterTypes". Each
%                           row of the cell array can contain an array of
%                           parameters.
%   verbosity           :   set to 1 to see the filtered and original eye
%                           position data. set to 0 for no feedback.
%   plotAxis            :   axes handle for giving feedback. if not
%                           provided or empty, a new figure is created. 
%
%   Example usage: 
%   
% parameterStructure.overwrite = 0;
% parameterStructure.maxGapDurationMs = 10; %ms
% parameterStructure.filterTypes = {@medfilt1, @sgolayfilt};
% parameterStructure.filterParameters = {11,[3 15]}; % 11 is for medfilt1, [3 15] 
%                     is for sgolayfilt (3 is degree of poly, 15 size of kernel)
% FilterEyePosition('myfile.mat',parameterStructure);
% 
% OR
%
% inputArray = [eyePosition time];
% filteredPosition = FilterEyePosition(inputArray,parameterStructure);
%
%

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

%% Check |parameterStructure|
if ~isfield(parametersStructure,'overwrite')
    overwrite = 0; 
else
    overwrite = parametersStructure.overwrite;
end

if ~isfield(parametersStructure,'maxGapDurationMs')
    maxGapDurationMs = 10; 
else
    maxGapDurationMs = parametersStructure.maxGapDurationMs;
end

if ~isfield(parametersStructure,'filterTypes')
    filterTypes = {@sgolayfilt}; 
else
    filterTypes = parametersStructure.filterTypes;
end

if ~isfield(parametersStructure,'filterParameters')
    filterParameters = {[3 15]};  
else
    filterParameters = parametersStructure.filterParameters;
end

if ~isfield(parametersStructure,'verbosity')
    verbosity = 0;  
else
    verbosity = parametersStructure.verbosity;
end

if ~isfield(parametersStructure,'plotAxis')
    plotAxis = [];  
else
    plotAxis = parametersStructure.plotAxis;
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


%% Do filtering.
% note: although some filters can be applied to 2D arrays directly (e.g., 
% sgolayfilt), to preserve generality, we will filter each position
% column separately.

% preallocate memory for the filtered position
filteredEyePosition = nan(size(eyePositionTraces));

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
            ['currentFilter(eyePositionTraces(:,j)' parameterStr ')']);
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

%% Remove the rest.
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
    data.filteredEyePosition = filteredEyePosition;
    data.filterParametersStructure = parametersStructure;
    save(outputFilePath,'-struct','data');
end

%% Give feedback if user requested.
if verbosity
   if ishandle(plotAxis)
       axes(plotAxis);
   else
       figure(1453);
       cla;
       axes(gca);
   end
   plot(timeArray,eyePositionTraces); hold on;
   plot(timeArray,filteredEyePosition,'LineWidth',2);
   set(gca,'Fontsize',14);
   xlabel('Time (sec)');
   ylabel('Eye position');
end





