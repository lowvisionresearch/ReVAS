function [newEyePositionTraces, outputFilePath, params] = ...
    ReReference(positionArgument, localRefArgument, globalRefArgument, ...
    parametersStructure)
%REREFERENCE adds offsets to eye position traces so that all positions are
%   based on a global reference frame. 
%
%   'positionArgument' can be either an Nx2 array of eyePositionTraces, or
%   a full path to a file containing eyePositionTraces. 'localRefArgument'
%   can be either a 2D array representing the local ref corresponding to
%   'positionArgument', or a full path to a file containing 'localRef'.
%   'globalRefArgument' can be either a 2D array representing the global
%   ref, or a full path to a file containing 'globalRef'.
%   'parametersStructure' is a structure specifying the parameters of the
%   re-referencing process.
%
%   Fields of the 'parametersStructure' structure
%   --------------------------------
%   overwrite               : set to 1 if you want to overwrite the
%                             existing output file. Relevant only when 
%                             'positionArgument' is a filename. It is
%                             ignored if 'positionArgument' is actual eye
%                             position traces.
%   fixTorsion              : set to 1 if you want to improve
%                             cross-correlations by rotating the local ref
%                             with a range of tilts.
%   tilts                   : a 1D array with a set of tilt values. Default
%                             values are -5:1:5;
%   findPeakMethod          : set to 1 if you want to find the peak of raw
%                             cross-correlation map to localize local ref
%                             on global ref. set to 2 if you want to enable
%                             subtraction of gaussian-filtered version of
%                             the cross-correlation map from itself before
%                             searching for the peak. Defaults to 2
%                             (time-consuming).
%   findPeakKernelSize      : size of the gaussian kernel for median
%                             filtering. defaults to 21. ignored if
%                             findPeakMethod is 1.
%   searchZone              : size of the search zone for the peak, in
%                             terms of fraction of the cross-correlation
%                             map. e.g., a searchZone of 1 means, the
%                             entire map will be searched. 0.2 means that
%                             central (0.5-0.1=)0.4 to (0.5+0.1=)0.6 part
%                             of the map will be searched. defaults to 0.5.
%   enableVerbosity         : set to 1 to enable user
%   feedback. axesHandles   : axis handle to give feedback. Relevant only
%                             when 'enableVerbosity' is enabled. if this
%                             field does not exist or empty AND
%                             'enableVerbosity' is set to 1, a new figure
%                             is generated and feedback is given via that
%                             figure.
%



%% Handle misusage 
if nargin<3
    error('ReReference needs at least three four arguments.')
end

if nargin<4
    error('parametersStructure is not provided, re-referencing cannot proceed.');
end 

if ~isstruct(parametersStructure)
    error('''parametersStructure'' must be a struct.');
end

%% Set parameters to defaults if not specified.
if ~isfield(parametersStructure,'overwrite')
    overwrite = false;
else
    overwrite = parametersStructure.overwrite;
    if ~islogical(overwrite)
        error('overwrite must be a logical');
    end
end

if ~isfield(parametersStructure,'fixTorsion')
    fixTorsion = false;
    RevasWarning('using default parameter for fixTorsion', fixTorsion);
else
    fixTorsion = parametersStructure.fixTorsion;
    if ~islogical(fixTorsion)
        error('fixTorsion must be a logical');
    end
end

if ~isfield(parametersStructure,'tilts')
    tilts = -5:1:5;
    RevasWarning('using default parameter for tilts', parametersStructure);
else
    tilts = parametersStructure.tilts;
    if size(tilts, 2) == 1
        tilts = tilts';
    end
    if ~isnumeric(tilts) && size(tilts, 1) ~= 1
       error('tilts must be a vector of numbers'); 
    end
end



if ~isfield(parametersStructure,'findPeakMethod')
    findPeakMethod = 2;
else
    findPeakMethod = parametersStructure.findPeakMethod;
    if findPeakMethod ~= 1 && findPeakMethod ~= 2
        error('findPeakMethod must be 1 or 2');
    end
end


if ~isfield(parametersStructure,'findPeakKernelSize')
    findPeakKernelSize = 21;
else
    findPeakKernelSize = parametersStructure.findPeakKernelSize;
    if ~IsOddNaturalNumber(findPeakKernelSize)
        error('findPeakKernelSize must be an odd, natural number');
    end
end



if ~isfield(parametersStructure,'searchZone')
    searchZone = 0.5;
else
    searchZone = parametersStructure.searchZone;
    if ~IsRealNumber(searchZone) || searchZone < 0 || searchZone > 1
       error('searchZone must be a real number between 0 and 1 (inclusive)');
    end
end


if ~isfield(parametersStructure,'enableVerbosity')
    enableVerbosity = false;
else
    enableVerbosity = parametersStructure.enableVerbosity;
    if ~islogical(enableVerbosity)
        error('enableVerbosity must be a logical');
    end
end

if ~isfield(parametersStructure,'axesHandles')
    axesHandles = [];
else
    axesHandles = parametersStructure.axesHandles;
end

%% Handle |positionArgument| scenarios.
if ischar(positionArgument) % positionArgument is a file path
    outputFilePath = [positionArgument(1:end-4) '_reref.mat'];
    
    % Handle overwrite scenarios
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~overwrite && exist(outputFilePath, 'file')
        RevasWarning(['ReReference() did not execute because it would overwrite existing file. (' outputFilePath ')'], parametersStructure);
        return;
    else
        RevasWarning(['ReReference() is proceeding and overwriting an existing file. (' outputFilePath ')'], parametersStructure);  
    end
    
    % check if input file exists
    if ~exist(positionArgument,'file')
        error('eye position file does not exist!');
    end
    
    % load the data
    data = load(positionArgument,'eyePositionTraces','timeArray');
    eyePositionTraces = data.eyePositionTraces;

else % inputArgument is not a file path, but carries the eye position data.
    outputFilePath = [];
    
    eyePositionTraces = positionArgument;
    
end

%% Handle |localRefArgument| scenarios.
if ischar(localRefArgument) % localRefArgument is a file path

    % check if input file exists
    if ~exist(localRefArgument,'file')
        error('local ref file does not exist!');
    end
    
    try
        % load the localRef
        load(localRefArgument,'refFrame');
        localRef = refFrame*255;
    catch
        try
            % maybe this is a coarse ref file
            load(localRefArgument,'coarseRefFrame');
            localRef = coarseRefFrame*255;
        catch
            % maybe this is an image file
            localRef = im2double(imread(localRefArgument))*255;
        end
    end

else % localRefArgument is not a file path, but carries the localRef.
    localRef = localRefArgument;
end

%% Handle |globalRefArgument| scenarios.
if ischar(globalRefArgument) % globalRefArgument is a file path

    % check if input file exists
    if ~exist(globalRefArgument,'file')
        error('global ref file does not exist!');
    end
    
    try
        % load the globalRef
        load(globalRefArgument,'globalRef');
    catch
        % maybe this is an image file
        globalRef = imread(globalRefArgument);
    end

else % localRefArgument is not a file path, but carries the localRef.
    globalRef = globalRefArgument;
end

%% Re-referencing done here.
% compute ref centers
localCenter = size(localRef)/2;
globalCenter = size(globalRef)/2;

% remove black edges around the refs, if any
localRef = PadNoise(localRef);
globalRef = PadNoise(globalRef);
    
% prepare structure for Localize sub-function
params.findPeakMethod = findPeakMethod;
params.findPeakKernelSize = findPeakKernelSize;
params.searchZone = searchZone;
params.tilts = tilts;

% localize local ref on global ref
if ~fixTorsion
    [yOffset, xOffset, peakValue, c, ~, ~] = Localize(localRef,globalRef,params);
else
    [localRef, bestTilt, yOffset, xOffset, peakValue, c] = ...
        SolveTiltIssue(localRef,globalRef,params);
end



% adjust eye position traces based on the estimated offsets
offsetBetweenLocalAndGlobal = [xOffset yOffset ];
newEyePositionTraces = eyePositionTraces...
    + repmat(offsetBetweenLocalAndGlobal,length(eyePositionTraces),1);

%% Save re-referenced data.
if ~isempty(outputFilePath) && overwrite
    data.eyePositionTraces = newEyePositionTraces;
    data.offsetBetweenLocalAndGlobal = offsetBetweenLocalAndGlobal;
    data.peakValue = peakValue;
    data.parametersStructure = parametersStructure;
    if exist('bestTilt','var')
        data.bestTilt = bestTilt;
    end
    save(outputFilePath,'-struct','data');
end

%% Give feedback if user requested.
if enableVerbosity
    
    if ishandle(axesHandles)
        axes(axesHandles)
    else
        figure;
        axesHandles = gca;
    end
    
    % the following hack is needed just for plotting!
    
    % pre-padding when offsets are negative
    if xOffset < 1 
        prepadX = abs(xOffset);
        xOffset = 1;
    else
        prepadX = 0;
    end
    
    if yOffset < 1
        prepadY = abs(yOffset);
        yOffset = 1;
    else
        prepadY = 0;
    end
    tempGlobal = padarray(globalRef,[prepadX prepadY],'pre');
    
    
    % post-padding when the local ref is bigger than the pre-padded global
    % ref
    if abs(xOffset) + prepadX + size(localRef,2) > size(tempGlobal,2) 
        padX = abs(xOffset) + prepadX + size(localRef,2) - size(tempGlobal,2) + 1;
    else
        padX = 0;
    end
    
    if abs(yOffset) + prepadY + size(localRef,1) > size(tempGlobal,1)
        padY = abs(yOffset) + prepadY + size(localRef,1) - size(tempGlobal,1) + 1;
    else
        padY = 0;
    end
    
    if padX < 0 
        padX = 0;
    end
    if padY < 0 
        padY = 0;
    end

    tempGlobal = padarray(tempGlobal,[padY padX],'post');
    % end hack
    
    
    tempGlobal = double(tempGlobal);
    yind = yOffset:yOffset+size(localRef,1)-1;
    xind = xOffset:xOffset+size(localRef,2)-1;
    tempGlobal(yind,xind) = (tempGlobal(yind,xind) + double(localRef))/2;
    imshow(uint8(tempGlobal));
    
end

%% localizer function
function [yoffset, xoffset, maxVal, c, ypeak, xpeak] = Localize(inp,ref, params)
% does the cross-correlation and locates the peak. Handles scenarios where
% inp is larger than the ref.

try
    N = [0 0];
    c = normxcorr2(inp,ref);
catch err1 %#ok<*NASGU>
    
    % probably complaining about size, so zeropad the ref and see how it
    % goes. don't forget to control for the additional offsets this causes.
    N = size(inp)-size(ref);
    N(N<0) = 0;
    
    tempRef = padarray(ref,N,'post');
    try
        c = normxcorr2(inp,tempRef);
        
    catch err2
        % probably complaining about size
        yoffset = 0; xoffset = 0; maxVal = 0;
        return;
    end
end

[ypeak, xpeak, maxVal] = FindPeak(c, params);
yoffset = ypeak-size(inp,1);
xoffset = xpeak-size(inp,2);

%% peakfinder.  
function [ypeak, xpeak, maxVal] = FindPeak(c, params)

if params.findPeakMethod == 2
    tempC = c - imgaussfilt(c,params.findPeakKernelSize);
else
    tempC = c;
end

st = round(size(tempC)*(.5-params.searchZone/2));
en = round(size(tempC)*(.5+params.searchZone/2));
tempC = tempC(st(1):en(1),...
          st(2):en(2));
maxVal = max(tempC(:));
[ypeak, xpeak] = find(tempC == maxVal);

ypeak = ypeak + st(1) - 1;
xpeak = xpeak + st(2) - 1;
maxVal = c(ypeak,xpeak);

%% tilt issue solver
function [inp, bestTilt, bestYOffset, bestXOffset, peakVal, c, ypeak, xpeak] ...
    = SolveTiltIssue(inp,ref,params)

tilts = params.tilts;
yOffset = nan(length(tilts),1);
xOffset = nan(length(tilts),1);
maxVal = nan(length(tilts),1);

% rotate the input image, crosscorrelate with ref, and find the peak.
% Across all tilts, find the best one which results in the max peak.
% Also rotate the input image based on the bestTilt.
for j=1:length(tilts)
    rotated = imrotate(inp,tilts(j),'bilinear','crop');
    rotated = PadNoise(rotated);
    [yOffset(j), xOffset(j), maxVal(j)] = Localize(rotated,ref, params);
end

% now interpolate the peak values
newTilts = (min(tilts):0.05:max(tilts));
peaks = interp1(tilts,maxVal,newTilts,'pchip');

% find best tilt
bestTilt = newTilts(peaks == max(peaks));

% rotate for the best tilt
inp = imrotate(inp,bestTilt,'bilinear','crop');
inp = uint8(PadNoise(inp));

% localize
[bestYOffset, bestXOffset, peakVal, c, ypeak, xpeak] = Localize(inp,ref, params);

%% add noise to zero-padded regions due to tilt
function output = PadNoise(input)

input = double(input);
padIndices = input < 1.5;
noiseFrame = padIndices.*(rand(size(padIndices))*20 + mean(input(:)));
output = double(input) + noiseFrame;
