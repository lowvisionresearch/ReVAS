function [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure] = ...
    StripAnalysis(inputVideo, referenceFrame, parametersStructure)
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
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite                       : set to true to overwrite existing files.
%                                     Set to false to abort the function call if the
%                                     files already exist. (default false)
%   enableVerbosity                 : set to true to report back plots during execution.
%                                     (default false)
%   badFrames                       : vector containing the frame numbers of
%                                     the blink frames. (default [])
%   stripHeight                     : strip height to be used for strip
%                                     analysis in pixels. (default 15)
%   stripWidth                      : strip width to be used for strip
%                                     analysis in pixels. Should be set to
%                                     the width of each frame. (default 488)
%   samplingRate                    : sampling rate of the video in Hz.
%                                     (default 540)
%   enableGaussianFiltering         : set to true to enable Gaussian filtering. 
%                                     Set to false to disable. (default
%                                     false)
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
%   maximumPeakRatio                : maximum peak ratio allowed between the
%                                     highest and second-highest peak in a 
%                                     correlation map--strips with positions
%                                     that have a peak ratio >
%                                     maximumPeakRatio will be discarded.
%                                     (relevant only when
%                                     enableGaussianFiltering is false)
%                                     (default 0.8)
%   minimumPeakThreshold            : the minimum value above which a peak
%                                     needs to be in order to be considered 
%                                     a valid correlation. (this applies
%                                     regardless of enableGaussianFiltering)
%                                     (default 0)
%   adaptiveSearch                  : set to true to perform search on
%                                     scaled down versions first in order
%                                     to potentially improve computation
%                                     time. (default false)
%   scalingFactor                   : the factor to scale down by.
%                                     (relevant only when adaptiveSearch is
%                                     true) (default 8)
%   searchWindowHeight              : the height of the search window to be
%                                     used for adaptive search in pixels.
%                                     (relevant only when adaptiveSearch is
%                                     true) (default 79)
%   enableSubpixelInterpolation     : set to true to estimate peak
%                                     coordinates to a subpixel precision
%                                     through interpolation. (default false)
%   subpixelInterpolationParameters : see below. (relevant only if
%                                     enableSubpixelInterpolation is true)
%   createStabilizedVideo           : set to true to create a stabilized
%                                     videos. (default false)
%   axesHandles                     : axes handle for giving feedback. if not
%                                     provided or empty, new figures are created.
%                                     (relevant only when enableVerbosity is true)
%   corrMethod                      : method to use for cross-correlation.
%                                     you can choose from 'normxcorr' for
%                                     matlab's built-in normxcorr2, 'mex'
%                                     for opencv's correlation, or 'fft'
%                                     for our custom-implemented fast
%                                     correlation method.
%   downSampleFactor                : only utilized for corrMethod of
%                                     'fft'. If > 1, the reference frame
%                                     and strips are shrunk by that factor
%                                     prior to correlation. If < 1, every 
%                                     other pixel of the reference frame
%                                     is kept (in a checkerboard-like 
%                                     pattern). (default 1)
%
%   -----------------------------------
%   Fields of the |subpixelInterpolationParameters|
%   -----------------------------------
%   neighborhoodSize                : the length of one of the sides of the
%                                     neighborhood area over which
%                                     interpolation is to be performed over
%                                     in pixels. (default 7)
%   subpixelDepth                   : the scaling of the desired level of
%                                     subpixel depth. (default 2)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'MyVid.avi';
%       load('MyVid_refFrame.mat');
%       parametersStructure.overwrite = true;
%       parametersStructure.enableVerbosity = true;
%       parametersStructure.stripHeight = 15;
%       parametersStructure.stripWidth = 488;
%       parametersStructure.samplingRate = 540;
%       parametersStructure.enableGaussianFiltering = false;
%       parametersStructure.maximumPeakRatio = 0.8;
%       parametersStructure.minimumPeakThreshold = 0;
%       parametersStructure.adaptiveSearch = false;
%       parametersStructure.enableSubpixelInterpolation = true;
%       parametersStructure.subpixelInterpolationParameters.neighborhoodSize
%           = 7;
%       parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
%           = 2;
%       parametersStructure.createStabilizedVideo = false;
%       [rawEyePositionTraces, usefulEyePositionTraces, timeArray, statisticsStructure] = ...
%           StripAnalysis(inputVideo, referenceFrame, parametersStructure);

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

% Default frame rate if a matrix representation of the video passed in.
% Users may also specify custom frame rate via parametersStructure.
if ~writeResult && ~isfield(parametersStructure, 'FrameRate')
    parametersStructure.FrameRate = 30;
    RevasWarning('using default parameter for FrameRate', parametersStructure);
end

% If referenceFrame is a character array, then a path was passed in.
if ischar(referenceFrame)
    % Reference Frame Path is needed because it is written to the file in
    % the end.
    referenceFramePath = referenceFrame;
 
    success = false;
    data = load(referenceFramePath, 'refFrame');
    if isfield(data, 'refFrame')
        referenceFrame = data.refFrame;
        success = true;
    end
    
    data = load(referenceFramePath, 'coarseRefFrame');
    if isfield(data, 'coarseRefFrame')
        referenceFrame = data.coarseRefFrame;
        success = true;
    end
    
    if ~success
        error(['No reference frame could be loaded from ' referenceFramePath]);
    end
    
    clear data
    clear success
else
    referenceFramePath = '';
end
if ~ismatrix(referenceFrame)
    error('Invalid Input for referenceFrame (it was not a 2D array)');
end

% Identify which frames are bad frames
% The filename is unknown if a raw array was passed in.
if ~isfield(parametersStructure, 'badFrames')
    if writeResult
        blinkFramesPath = Filename(inputVideo, 'blink');
    else
        blinkFramesPath = fullfile(pwd, '.blinkframes.mat');
    end
    
    try
        load(blinkFramesPath, 'badFrames');
    catch
        badFrames = [];
    end
else
   badFrames = parametersStructure.badFrames;
end

if ~isfield(parametersStructure, 'stripHeight')
    parametersStructure.stripHeight = 15;
    RevasMessage('using default parameter for stripHeight');
else
    if ~IsNaturalNumber(parametersStructure.stripHeight)
        error('stripHeight must be a natural number');
    end
end

if ~isfield(parametersStructure, 'stripWidth') && writeResult
    reader = VideoReader(inputVideo);
    parametersStructure.stripWidth = reader.Width;
    RevasMessage('using default parameter for stripWidth');
elseif ~isfield(parametersStructure, 'stripWidth') && ~writeResult
    parametersStructure.stripWidth = size(inputVideo, 2);
    RevasMessage('using default parameter for stripWidth');
else
    if ~IsNaturalNumber(parametersStructure.stripWidth)
        error('stripWidth must be a natural number');
    end
end

if ~isfield(parametersStructure, 'samplingRate')
    parametersStructure.samplingRate = 540;
    RevasMessage('using default parameter for samplingRate');
else
    if ~IsNaturalNumber(parametersStructure.samplingRate)
        error('samplingRate must be a natural number');
    end
end

if ~isfield(parametersStructure, 'enableGaussianFiltering')
    parametersStructure.enableGaussianFiltering = false;
else
    if ~islogical(parametersStructure.enableGaussianFiltering)
        error('enableGaussianFiltering must be a logical');
    end
end

if ~isfield(parametersStructure, 'maximumSD')
    parametersStructure.maximumSD = 10;
    RevasMessage('using default parameter for maximumSD');
else
    if ~IsPositiveRealNumber(parametersStructure.maximumSD)
        error('maximumSD must be a positive, real number');
    end
end

if ~isfield(parametersStructure, 'SDWindowSize')
    parametersStructure.SDWindowSize = 25;
    RevasMessage('using default parameter for SDWindowSize');
else
    if ~IsNaturalNumber(parametersStructure.SDWindowSize)
        error('SDWindowSize must be a natural number');
    end
end

if ~isfield(parametersStructure, 'maximumPeakRatio')
    parametersStructure.maximumPeakRatio = 0.8;
    RevasMessage('using default parameter for maximumPeakRatio');
else
    if ~IsPositiveRealNumber(parametersStructure.maximumPeakRatio)
        error('maximumPeakRatio must be a positive, real number');
    end
end

if ~isfield(parametersStructure, 'minimumPeakThreshold')
    parametersStructure.minimumPeakThreshold = 0;
    RevasMessage('using default parameter for minimumPeakThreshold');
else
    if ~IsNonNegativeRealNumber(parametersStructure.minimumPeakThreshold)
        error('minimumPeakThreshold must be a non-negative, real number');
    end
end

if ~isfield(parametersStructure, 'adaptiveSearch')
    parametersStructure.adaptiveSearch = false;
else
    if ~islogical(parametersStructure.adaptiveSearch)
        error('adaptiveSearch must be a logical');
    end
end

if ~isfield(parametersStructure, 'scalingFactor')
    parametersStructure.scalingFactor = 10;
    RevasMessage('using default parameter for scalingFactor');
else
    if ~IsPositiveRealNumber(parametersStructure.scalingFactor)
        error('scalingFactor must be a positive, real number');
    end
end

if ~isfield(parametersStructure, 'searchWindowHeight')
    parametersStructure.searchWindowHeight = 79;
    RevasMessage('using default parameter for searchWindowHeight');
else
    if ~IsNaturalNumber(parametersStructure.searchWindowHeight)
        error('searchWindowHeight must be a natural number');
    end
end

if ~isfield(parametersStructure, 'enableSubpixelInterpolation')
    parametersStructure.enableSubpixelInterpolation = false;
else
    if ~islogical(parametersStructure.enableSubpixelInterpolation)
        error('enableSubpixelInterpolation must be a logical');
    end
end

if ~isfield(parametersStructure, 'subpixelInterpolationParameters')
   parametersStructure.subpixelInterpolationParameters = struct;
   parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
   parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
   RevasMessage('using default parameter for subpixelInterpolationParameters');
else
    if ~isstruct(parametersStructure.subpixelInterpolationParameters)
       error('subpixelInterpolationParameters must be a struct');
    else
       if ~isfield(parametersStructure.subpixelInterpolationParameters, 'neighborhoodSize')
           parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
           RevasMessage('using default parameter for neighborhoodSize');
       else
           parametersStructure.subpixelInterpolationParameters.neighborhoodSize = parametersStructure.subpixelInterpolationParameters.neighborhoodSize;
           if ~IsNaturalNumber(parametersStructure.subpixelInterpolationParameters.neighborhoodSize)
               error('subpixelInterpolationParameters.neighborhoodSize must be a natural number');
           end
       end
       if ~isfield(parametersStructure.subpixelInterpolationParameters, 'subpixelDepth')
           parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
           RevasMessage('using default parameter for subpixelDepth');
       else
           parametersStructure.subpixelInterpolationParameters.subpixelDepth = parametersStructure.subpixelInterpolationParameters.subpixelDepth;
           if ~IsPositiveRealNumber(parametersStructure.subpixelInterpolationParameters.subpixelDepth)
               error('subpixelInterpolationParameters.subpixelDepth must be a positive, real number');
           end
       end
    end
end

if ~isfield(parametersStructure, 'enableVerbosity')
   parametersStructure.enableVerbosity = false; 
else
   if ~islogical(parametersStructure.enableVerbosity)
        error('enableVerbosity must be a logical');
   end
end

if ~isfield(parametersStructure, 'createStabilizedVideo')
    parametersStructure.createStabilizedVideo = false;
else
    if ~islogical(parametersStructure.createStabilizedVideo)
        error('createStabilizedVideo must be a logical');
    end
end

if ~isfield(parametersStructure, 'enableGPU')
    parametersStructure.enableGPU = false;
else
    if ~islogical(parametersStructure.enableGPU)
        error('enableGPU must be a logical');
    end
end
parametersStructure.enableGPU = (gpuDeviceCount > 0) & parametersStructure.enableGPU;

if ~isfield(parametersStructure, 'corrMethod')
    parametersStructure.corrMethod = 'mex';
end

if ~isfield(parametersStructure, 'downSampleFactor')
    parametersStructure.downSampleFactor = 1;
end

%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputVideo, 'usefultraces', parametersStructure.samplingRate);

    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
        RevasMessage(['StripAnalysis() did not execute because it would overwrite existing file. (' outputFilePath ')'], parametersStructure);
        rawEyePositionTraces = [];
        usefulEyePositionTraces = [];
        timeArray = [];
        statisticsStructure = struct();
        return;
    else
        RevasMessage(['StripAnalysis() is proceeding and overwriting an existing file. (' outputFilePath ')'], parametersStructure);  
    end
end

%% Preallocation and variable setup

if writeResult
    reader = VideoReader(inputVideo);
    numberOfFrames = reader.FrameRate * reader.Duration;
    height = reader.Height;
    parametersStructure.FrameRate = reader.FrameRate;
else
    [height, ~, numberOfFrames] = size(inputVideo);
end
    
stripsPerFrame = round(parametersStructure.samplingRate / parametersStructure.FrameRate);
distanceBetweenStrips = (height - parametersStructure.stripHeight)...
    / (stripsPerFrame - 1);
numberOfStrips = stripsPerFrame * numberOfFrames;

% two columns for horizontal and vertical movements
rawEyePositionTraces = NaN(numberOfStrips, 2);

% arrays for peak and second highest peak values
peakValueArray = zeros(numberOfStrips, 1);
secondPeakValueArray = zeros(numberOfStrips, 1);

% array for standard deviations (used for gaussian peaks approach)
standardDeviationsArray = NaN(numberOfStrips, 2);

% array for search windows
searchWindowsArray = NaN(numberOfStrips, 2);

%% Populate time array
timeArray = (1:numberOfStrips)' / parametersStructure.samplingRate;

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end
isSetView = true;

%% Normalized cross-correlate each strip
% Note that calculation for each array value does not end with this loop,
% the logic below the loop in this section perform remaining operations on
% the values but are done outside of the loop in order to take advantage of
% vectorization (that is, if verbosity is not enabled since if it was, then
% these operations must be computed immediately so that the correct eye
% trace values can be plotted as early as possible).

if parametersStructure.enableGPU
    referenceFrame = gpuArray(referenceFrame);
end

currFrameNumber = 0;

if writeResult
    reader = VideoReader(inputVideo);
else
    
end

% Variable for fft corrmethod
cache = struct; 
cacheAdaptive = struct; 

% Variables for adaptive search:
loc = (height / stripsPerFrame) / 2;
% remember the last 4 velocities
historyCapacity = 4;
% velHistory will act as a circular queue, with the next to be deleted
% marked by historyIndex.
historyIndex = 2;
velHistory = [height / stripsPerFrame];

for stripNumber = 1:numberOfStrips

    if ~abortTriggered
        
        gpuTask = getCurrentTask;

        % Note that only one core should use the GPU at a time.
        % i.e. when processing multiple videos in parallel, only one should
        % use GPU.
        
        rowNumber = floor(mod(stripNumber - 1, stripsPerFrame) * distanceBetweenStrips + 1);
        % Edge case for if there is only strip per frame.
        if isnan(rowNumber) && stripsPerFrame == 1
            rowNumber = 1;
        end
        colNumber = 1;
        frameNumber = floor((stripNumber-1) / stripsPerFrame + 1);

        if frameNumber > currFrameNumber
            currFrameNumber = frameNumber;
            
            if writeResult
                frame = readFrame(reader);
                if ndims(frame) == 3
                    frame = rgb2gray(frame);
                end
            else
                frame = inputVideo(1:end, 1:end, currFrameNumber);
            end
        end

        if ismember(frameNumber, badFrames)
            rawEyePositionTraces(stripNumber,:) = [NaN NaN];
            peakValueArray(stripNumber) = NaN;
            secondPeakValueArray(stripNumber) = NaN;
            continue;
        end

        rowEnd = rowNumber + parametersStructure.stripHeight - 1;
        columnEnd = colNumber + parametersStructure.stripWidth - 1;
        
        if ~parametersStructure.enableGPU
            strip = frame(rowNumber:rowEnd, colNumber:columnEnd);
        else
            strip = gpuArray(frame(rowNumber:rowEnd, colNumber:columnEnd));
        end
        
        parametersStructure.stripNumber = stripNumber;  
        parametersStructure.stripsPerFrame = stripsPerFrame;
        
        if parametersStructure.adaptiveSearch

            % Cut out a smaller search window from correlation,
            % centered around loc, and searchWindowHeight tall.
            upperBound = floor(min(max(1, loc - parametersStructure.searchWindowHeight/2)));
            lowerBound = upperBound + parametersStructure.searchWindowHeight - 1;
            
            if lowerBound > size(referenceFrame, 1)
               lowerBound = size(referenceFrame, 1);
               upperBound = lowerBound - parametersStructure.searchWindowHeight + 1;
            end
            
            if isequal(parametersStructure.corrMethod, 'normxcorr')
                adaptedCorrelationMap = normxcorr2(strip, referenceFrame);
            elseif isequal(parametersStructure.corrMethod, 'mex') && ...
                ~parametersStructure.enableGPU
                adaptedCorrelationMap = matchTemplateOCV( ...
                    strip, ...
                    referenceFrame(upperBound:lowerBound, 1:end));
            elseif isequal(parametersStructure.corrMethod, 'mex') && ...
                parametersStructure.enableGPU
                adaptedCorrelationMap = matchTemplateOCV_GPU( ...
                    strip, ...
                    referenceFrame(upperBound:lowerBound, 1:end));
            elseif isequal(parametersStructure.corrMethod, 'fft')
                [adaptedCorrelationMap, cacheAdaptive] = FastStripCorrelation( ...
                    strip, ...
                    referenceFrame, ...
                    cacheAdaptive, ...
                    parameterStructure.downSampleFactor, ...
                    parametersStructure.enableGPU);
            end
            
            % Try to use adapted version of correlation map.
            [xPeak, yPeak, peakValue, secondPeakValue] = ...
                FindPeak(adaptedCorrelationMap, parametersStructure);

            % See if adapted result is acceptable or not.
            if ~parametersStructure.enableGaussianFiltering && ...
                    (peakValue <= 0 || secondPeakValue <= 0 ...
                    || secondPeakValue / peakValue > parametersStructure.maximumPeakRatio ...
                    || peakValue < parametersStructure.minimumPeakThreshold)
                
                isAcceptable = false;
                
            elseif parametersStructure.enableGaussianFiltering % TODO, need to test.
                % Middle row SDs in column 1, Middle column SDs in column 2.
                middleRow = ...
                    adaptedCorrelationMap(max(ceil(yPeak-...
                        SDWindowSize/2/scalingFactor),...
                        1): ...
                    min(floor(yPeak+...
                        SDWindowSize/2/scalingFactor),...
                        size(adaptedCorrelationMap,1)), ...
                        floor(xPeak));
                middleCol = ...
                    adaptedCorrelationMap(floor(yPeak), ...
                    max(ceil(xPeak-...
                        SDWindowSize/2/scalingFactor),...
                        1): ...
                    min(floor(xPeak+SDWindowSize/2/scalingFactor),...
                        size(adaptedCorrelationMap,2)))';
                
                rowFit = fit(((1:size(middleRow,1))-ceil(size(middleRow,1)/2))', middleRow, 'gauss1');
                colFit = fit(((1:size(middleCol,1))-ceil(size(middleCol,1)/2))', middleCol, 'gauss1');
                
                if rowFit.c1 > parametersStructure.maximumSD || ...
                    colFit.c1 > parametersStructure.maximumSD || ...
                    peakValue <= 0 || ...
                    peakValue < parametersStructure.minimumPeakThreshold
                
                    isAcceptable = false;
                    
                else
                    isAcceptable = true;
                end
  
                clear rowFit;
                clear colFit;
            else
                isAcceptable = true;
            end
            
            correlationMap = adaptedCorrelationMap;
            searchWindowsArray(stripNumber,:) = [upperBound lowerBound];
        end
        
        if ~parametersStructure.adaptiveSearch || ...
                ~isAcceptable

            % Either adaptive search failed, the result was unacceptable, or we
            % didn't want to use adaptive search in the first place.
            % So we use the full correlation map.
            % It failed or was unacceptable, so use full correlation map.
            if isequal(parametersStructure.corrMethod, 'normxcorr')
                correlationMap = normxcorr2(strip, referenceFrame);
            elseif isequal(parametersStructure.corrMethod, 'mex') && ...
                    ~parametersStructure.enableGPU
                correlationMap = matchTemplateOCV(strip, referenceFrame);
            elseif isequal(parametersStructure.corrMethod, 'mex') && ...
                parametersStructure.enableGPU
                correlationMap = matchTemplateOCV_GPU(strip, referenceFrame);
            elseif isequal(parametersStructure.corrMethod, 'fft')
                [correlationMap, cache] = FastStripCorrelation( ...
                    strip, ...
                    referenceFrame, ...
                    cache, ...
                    parameterStructure.downSampleFactor, ...
                    parametersStructure.enableGPU);
            end
            [xPeak, yPeak, peakValue, secondPeakValue] = ...
                FindPeak(correlationMap, parametersStructure);

            searchWindowsArray(stripNumber,:) = [NaN NaN];
            
            upperBound = 1;
        end

        % 2D Interpolation if enabled
        if parametersStructure.enableSubpixelInterpolation
            [interpolatedPeakCoordinates, statisticsStructure.errorStructure] = ...
                Interpolation2D(correlationMap, [yPeak, xPeak], ...
                parametersStructure.subpixelInterpolationParameters);

            xPeak = interpolatedPeakCoordinates(2);
            yPeak = interpolatedPeakCoordinates(1);      
        end

        % If GPU was used, transfer peak values and peak locations
        if parametersStructure.enableGPU
            xPeak = gather(xPeak, gpuTask.ID);
            yPeak = gather(yPeak, gpuTask.ID);
            peakValue = gather(peakValue, gpuTask.ID);
            secondPeakValue = gather(secondPeakValue, gpuTask.ID);
        end

        if parametersStructure.enableGaussianFiltering
            % Fit a gaussian in a pixel window around the identified peak.
            % The pixel window is of size |parametersStructure.SDWindowSize|
            %
            % Take the middle row and the middle column, and fit a one-dimensional
            % gaussian to both in order to get the standard deviations.
            % Store results in statisticsStructure for choosing bad frames
            % later.

            % Middle row SDs in column 1, Middle column SDs in column 2.
            middleRow = ...
                correlationMap(max(ceil(yPeak-parametersStructure.SDWindowSize/2), 1): ...
                min(floor(yPeak+parametersStructure.SDWindowSize/2), size(correlationMap,1)), ...
                floor(xPeak));
            middleCol = ...
                correlationMap(floor(yPeak), ...
                max(ceil(xPeak-parametersStructure.SDWindowSize/2), 1): ...
                min(floor(xPeak+parametersStructure.SDWindowSize/2), size(correlationMap,2)))';
            fitOutput = fit(((1:size(middleRow,1))-ceil(size(middleRow,1)/2))', middleRow, 'gauss1');
            standardDeviationsArray(stripNumber, 1) = fitOutput.c1;
            fitOutput = fit(((1:size(middleCol,1))-ceil(size(middleCol,1)/2))', middleCol, 'gauss1');
            standardDeviationsArray(stripNumber, 2) = fitOutput.c1;
            clear fitOutput;
        end

        % If these peaks are in terms of an adapted correlation map, restore it
        % back to in terms of the full map.
        yPeak = yPeak + upperBound - 1;
        rawEyePositionTraces(stripNumber,:) = [xPeak yPeak];
        
        peakValueArray(stripNumber) = peakValue;
        secondPeakValueArray(stripNumber) = secondPeakValue;
        
        % Update adaptive search variables for next iteration.
        
        % loc was our guess for yPeak. Adjust velocity accordingly.
        % (This is how much we should have jumped from the previous yPeak
        % to land precisely on the correct place.)
        prevIndex = mod(historyIndex-2, historyCapacity) + 1;
        velHistory(prevIndex) = velHistory(prevIndex) + yPeak - loc;
        
        % Replace the oldest velocity with the current average velocity.
        velHistory(historyIndex) = mean(velHistory);
        
        % Advance to the next loc, based upon vel.
        % (Wrapping back to the top of the frame as necessary.)
        loc = yPeak + velHistory(historyIndex);
        if mod(stripNumber, stripsPerFrame) == 0
            loc = loc - size(referenceFrame, 1);
        end
        historyIndex = mod(historyIndex, historyCapacity) + 1;
        
        % We must subtract back out the expected strip coordinates in order
        % to obtain the net movement (the net difference between no
        % movement and the movement that was observed).
        rawEyePositionTraces(stripNumber,1) = ...
            rawEyePositionTraces(stripNumber,1) - colNumber;
        rawEyePositionTraces(stripNumber,2) = ...
            rawEyePositionTraces(stripNumber,2) - rowNumber;
        
        % Show surface plot for this correlation if verbosity enabled
        if parametersStructure.enableVerbosity
            if parametersStructure.enableGPU
                correlationMap = gather(correlationMap, gpuTask.ID);
            end
            if isfield(parametersStructure, 'axesHandles')
                axes(parametersStructure.axesHandles(1)); %#ok<*LAXES>
                cla;
                colormap(parametersStructure.axesHandles(1), 'default');
                if isSetView
                    view(3)
                    isSetView = false;
                end
            else
                figure(1);
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
        if parametersStructure.enableVerbosity

            % Adjust for padding offsets added by normalized cross-correlation.
            % If we enable verbosity and demand that we plot the points as we
            % go, then adjustments must be made here in order for the plot to
            % be interpretable.
            % Therefore, we will only perform these same operations after the
            % loop to take advantage of vectorization only if they are not
            % performed here, namely, if verbosity is not enabled and this
            % if statement does not execute.
            if ~isequal(parametersStructure.corrMethod, 'fft')
                rawEyePositionTraces(stripNumber,2) = ...
                    rawEyePositionTraces(stripNumber,2) - (parametersStructure.stripHeight - 1);
                rawEyePositionTraces(stripNumber,1) = ...
                    rawEyePositionTraces(stripNumber,1) - (parametersStructure.stripWidth - 1);
            end
            
            % Only scale up if downSampleFactor is > 1, since this means it
            % was shrunk during correlation.
            % If downSampleFactor was < 1, the images were thrown against a
            % bernoulli mask, but remained the same overall dimension.
            if parametersStructure.downSampleFactor > 1 && isequal(parametersStructure.corrMethod, 'fft')
               rawEyePositionTraces(stripNumber, :) = rawEyePositionTraces(stripNumber, :) .* parametersStructure.downSampleFactor; 
            end

            % We must subtract back out the expected strip coordinates in order
            % to obtain the net movement (the net difference between no
            % movement and the movement that was observed).
            rawEyePositionTraces(stripNumber,1) = ...
                rawEyePositionTraces(stripNumber,1) - (parametersStructure.stripWidth - 1);

            % Negate eye position traces to flip directions.
            rawEyePositionTraces(stripNumber,:) = -rawEyePositionTraces(stripNumber,:);

            if isfield(parametersStructure, 'axesHandles')
                axes(parametersStructure.axesHandles(2));
                colormap(parametersStructure.axesHandles(2), 'default');
            else
                figure(2);
            end
            plot(timeArray, rawEyePositionTraces);
            title('Raw Eye Position Traces');
            xlabel('Time (sec)');
            ylabel('Eye Position Traces (pixels)');
            legend('show');
            legend('Horizontal Traces', 'Vertical Traces');
        end
    end
end

%% Adjust for padding offsets added by normalized cross-correlation.
% Do this after the loop to take advantage of vectorization
% Only run this section if verbosity was not enabled. If verbosity was
% enabled, then these operations were already performed for each point
% before it was plotted to the eye traces graph. If verbosity was not
% enabled, then we do it now in order to take advantage of vectorization.
if ~parametersStructure.enableVerbosity
    if ~isequal(parametersStructure.corrMethod, 'fft')
        rawEyePositionTraces(:,2) = ...
            rawEyePositionTraces(:,2) - (parametersStructure.stripHeight - 1);
        rawEyePositionTraces(:,1) = ...
            rawEyePositionTraces(:,1) - (parametersStructure.stripWidth - 1);
    end
    
    % Only scale up if downSampleFactor is > 1, since this means it
    % was shrunk during correlation.
    % If downSampleFactor was < 1, the images were thrown against a
    % checkboard mask, but remained the same overall dimension.
    if parametersStructure.downSampleFactor > 1 && isequal(parametersStructure.corrMethod, 'fft')
       rawEyePositionTraces = rawEyePositionTraces .* parametersStructure.downSampleFactor; 
    end

    % Negate eye position traces to flip directions.
    rawEyePositionTraces = -rawEyePositionTraces;
end

%% Populate statisticsStructure

statisticsStructure.peakValues = peakValueArray;
statisticsStructure.peakRatios = secondPeakValueArray ./ peakValueArray;
statisticsStructure.searchWindows = searchWindowsArray;
statisticsStructure.standardDeviations = standardDeviationsArray;

%% Populate usefulEyePositionTraces

if parametersStructure.enableGaussianFiltering
    % Criteria for gaussian filtering method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * after a guassian is fitted in a 25x25 pixel window around the
    %   identified peak, the standard deviation of the curve must be below
    %   a maximum threshold in both the horizontal and vertical axes.
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToKeep = (statisticsStructure.standardDeviations(:,1) <= parametersStructure.maximumSD)...
        & (statisticsStructure.standardDeviations(:,2) <= parametersStructure.maximumSD)...
        & (statisticsStructure.peakValues >= parametersStructure.minimumPeakThreshold);
else
    % Criteria for peak ratio method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * the ratio of the second peak to the first peak is smaller than a
    %   maximum threshold (they must be sufficiently distinct).
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToKeep = (statisticsStructure.peakRatios <= parametersStructure.maximumPeakRatio)...
        & (statisticsStructure.peakValues >= parametersStructure.minimumPeakThreshold);
end

% multiply each component by 1 to keep eyePositionTraces or by NaN to toss.
usefulEyePositionTraces = rawEyePositionTraces; % duplicate vector first
usefulEyePositionTraces(~eyeTracesToKeep,:) = NaN;

%% Plot Useful Eye Traces
if ~abortTriggered && parametersStructure.enableVerbosity
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(2));
        colormap(parametersStructure.axesHandles(2), 'gray');
    else
        figure(3);
    end
    plot(timeArray, usefulEyePositionTraces);
    title('Useful Eye Position Traces');
    xlabel('Time (sec)');
    ylabel('Eye Position Traces (pixels)');
    legend('show');
    legend('Horizontal Traces', 'Vertical Traces');
    
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(1));
    else
        figure(11);
    end
    cla;
    
    plot(timeArray, statisticsStructure.peakRatios); hold on;
    plot(timeArray, statisticsStructure.peakValues);
    title('Sample quality');
    xlabel('Time (sec)');
    legend('show');
    legend('Peak ratio', 'Peak value');
    ylim([0 1])
    view(2);
end

%% Plot stimuli on reference frame
if ~abortTriggered && parametersStructure.enableVerbosity
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(3));
        colormap(parametersStructure.axesHandles(3), 'gray');
    else
        figure(4);
    end
    
    imshow(referenceFrame);
    hold on;
    
    center = fliplr(size(referenceFrame)/2);
    positionsToBePlotted = repmat(center, length(usefulEyePositionTraces),1) + usefulEyePositionTraces;
    
    scatter(positionsToBePlotted(:,1), positionsToBePlotted(:,2), 'y', 'o' , 'filled');
    hold off;
end

%% Save to output mat file

if writeResult && ~abortTriggered
    
    try
        parametersStructure = rmfield(parametersStructure,'axesHandles'); 
        parametersStructure = rmfield(parametersStructure,'commandWindowHandle'); 
    catch
    end
    
    % Save under file labeled 'final'.
    if writeResult
        eyePositionTraces = usefulEyePositionTraces; 
        peakRatios = statisticsStructure.peakRatios;
        save(outputFilePath, 'eyePositionTraces', 'rawEyePositionTraces', ...
            'timeArray', 'parametersStructure', 'referenceFramePath', ...
            'peakRatios');
    end
end

%% Create stabilized video if requested
if ~abortTriggered && parametersStructure.createStabilizedVideo
    parametersStructure.positions = eyePositionTraces;
    parametersStructure.time = timeArray;
    StabilizeVideo(inputVideoPath, parametersStructure);
end
