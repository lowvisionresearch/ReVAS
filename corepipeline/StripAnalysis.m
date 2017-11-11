function [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure]...
    = StripAnalysis(videoInput, referenceFrame, parametersStructure)
%STRIP ANALYSIS Extract eye movements in units of pixels.
%   Cross-correlation of horizontal strips with a pre-defined
%   reference frame.

%% Set parameters to defaults if not specified.

inputVideoPath = '';
% If videoInput is a character array, then a path was passed in.
% Attempt to convert it to a 3D array.
if ischar(videoInput)
    inputVideoPath = videoInput;
    [videoInput, videoFrameRate] = VideoPathToArray(videoInput);
else
    % ASSUMPTION
    % If only a raw matrix is provided, then we will take the frame rate to
    % be 30.
    warning('A raw matrix was provided; assuming that frame rate is 30 fps.');
    videoFrameRate = 30;
end
if ndims(videoInput) ~= 3 && ndims(videoInput) ~= 4
    error('Invalid Input for videoInput (it was not a 3D or 4D array)');
end
% TODO not tested on color videos yet.
% Change 4D arrays to 3D by making video grayscale. Assumes 4D arrays are
% in format (x, y, time, color).
if ndims(videoInput) == 4
    numberOfFrames = size(videoInput, 3);
    newVideoInput = squeeze(videoInput(:,:,:,1));
    for frame = (1:numberOfFrames)
        frame3D = squeeze(videoInput(:,:,frame,:));
        newVideoInput(:,:,frame) = rgb2gray(frame3D);
    end
    videoInput = newVideoInput;
end

% If referenceFrame is a character array, then a path was passed in.
if ischar(referenceFrame)
    referenceFrame = importdata(referenceFrame);
end
if ~ismatrix(referenceFrame)
    error('Invalid Input for referenceFrame (it was not a 2D array)');
end

% Identify which frames are bad frames
% The filename may not exist if a raw array was passed in.
if ~isfield(parametersStructure, 'badFrames')
    nameEnd = inputVideoPath(1:size(inputVideoPath, 2)-4);
    blinkFramesPath = [nameEnd '_blinkframes.mat'];
    try
        load(blinkFramesPath, 'badFrames');
    catch
        badFrames = [];
    end
else
   badFrames = parametersStructure.badFrames;
end

if ~isfield(parametersStructure, 'stripHeight')
    stripHeight = 15;
else
    stripHeight = parametersStructure.stripHeight;
    if ~IsNaturalNumber(stripHeight)
        error('stripHeight must be a natural number');
    end
end

if ~isfield(parametersStructure, 'stripWidth')
    stripWidth = size(videoInput, 2);
else
    stripWidth = parametersStructure.stripWidth;
    if ~IsNaturalNumber(stripWidth)
        error('stripWidth must be a natural number');
    end
end

if ~isfield(parametersStructure, 'samplingRate')
    samplingRate = 540;
else
    samplingRate = parametersStructure.samplingRate;
    if ~IsNaturalNumber(samplingRate)
        error('samplingRate must be a natural number');
    end
end

if ~isfield(parametersStructure, 'enableGaussianFiltering')
    enableGaussianFiltering = true;
else
    enableGaussianFiltering = parametersStructure.enableGaussianFiltering;
    if ~islogical(enableGaussianFiltering)
        error('enableGaussianFiltering must be a logical');
    end
end

if enableGaussianFiltering
    if ~isfield(parametersStructure, 'maximumSD')
        maximumSD = 10;
    else
        maximumSD = parametersStructure.maximumSD;
        if ~IsPositiveRealNumber(maximumSD)
            error('maximumSD must be a positive, real number');
        end
    end
    if ~isfield(parametersStructure, 'SDWindowSize')
        SDWindowSize = 25;
    else
        SDWindowSize = parametersStructure.SDWindowSize;
        if ~IsNaturalNumber(SDWindowSize)
            error('SDWindowSize must be a natural number');
        end
    end
else
    if ~isfield(parametersStructure, 'maximumPeakRatio')
        maximumPeakRatio = 0.8;
    else
        maximumPeakRatio = parametersStructure.maximumPeakRatio;
        if ~IsPositiveRealNumber(maximumPeakRatio)
            error('maximumPeakRatio must be a positive, real number');
        end
    end
end

if ~isfield(parametersStructure, 'minimumPeakThreshold')
    minimumPeakThreshold = 0;
else
    minimumPeakThreshold = parametersStructure.minimumPeakThreshold;
    if ~IsPositiveRealNumber(minimumPeakThreshold)
        error('minimumPeakThreshold must be a positive, real number');
    end
end

if ~isfield(parametersStructure, 'adaptiveSearch')
    adaptiveSearch = false;
else
    adaptiveSearch = parametersStructure.adaptiveSearch;
    if ~islogical(adaptiveSearch)
        error('adaptiveSearch must be a logical');
    end
end

if adaptiveSearch
    if ~isfield(parametersStructure, 'scalingFactor')
        scalingFactor = 10;
    else
        scalingFactor = parametersStructure.scalingFactor;
        if ~IsPositiveRealNumber(scalingFactor)
            error('scalingFactor must be a positive, real number');
        end
    end
    if ~isfield(parametersStructure, 'searchWindowHeight')
        searchWindowHeight = 0.8;
    else
        searchWindowHeight = parametersStructure.searchWindowHeight;
        if ~IsNaturalNumber(searchWindowHeight)
            error('searchWindowHeight must be a natural number');
        end
    end
end

if ~isfield(parametersStructure, 'enableSubpixelInterpolation')
    enableSubpixelInterpolation = false;
else
    enableSubpixelInterpolation = parametersStructure.enableSubpixelInterpolation;
    if ~islogical(enableSubpixelInterpolation)
        error('enableSubpixelInterpolation must be a logical');
    end
end

if enableSubpixelInterpolation
    if ~isfield(parametersStructure, 'subpixelInterpolationParameters')
       subpixelInterpolationParameters = struct;
       subpixelInterpolationParameters.neighborhoodSize = 7;
       subpixelInterpolationParameters.subpixelDepth = 2;
    else
        if ~isstruct(parametersStructure.subpixelInterpolationParameters)
           error('subpixelInterpolationParameters must be a struct');
        else
           subpixelInterpolationParameters = parametersStructure.subpixelInterpolationParameters;
           if ~isfield(parametersStructure.subpixelInterpolationParameters, 'neighborhoodSize')
               subpixelInterpolationParameters.neighborhoodSize = 7;
           else
               subpixelInterpolationParameters.neighborhoodSize = parametersStructure.subpixelInterpolationParameters.neighborhoodSize;
               if ~IsNaturalNumber(subpixelInterpolationParameters.neighborhoodSize)
                   error('subpixelInterpolationParameters.neighborhoodSize must be a natural number');
               end
           end
           if ~isfield(parametersStructure.subpixelInterpolationParameters, 'subpixelDepth')
               subpixelInterpolationParameters.subpixelDepth = 2;
           else
               subpixelInterpolationParameters.subpixelDepth = parametersStructure.subpixelInterpolationParameters.subpixelDepth;
               if ~IsPositiveRealNumber(subpixelInterpolationParameters.subpixelDepth)
                   error('subpixelInterpolationParameters.subpixelDepth must be a positive, real number');
               end
           end
        end
    end
end

if ~isfield(parametersStructure, 'enableVerbosity')
   enableVerbosity = false; 
else
   enableVerbosity = parametersStructure.enableVerbosity;
   if ~islogical(enableVerbosity)
        error('enableVerbosity must be a logical');
   end
end

%% Handle overwrite scenarios.

rawOutputFileName = [inputVideoPath(1:end-4) '_' ...
    int2str(samplingRate) '_hz_raw'];
outputFileName = [inputVideoPath(1:end-4) '_' ...
    int2str(samplingRate) '_hz_final'];

if ~exist([rawOutputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['StripAnalysis() did not execute because it would overwrite existing file. (' rawOutputFileName ')'], parametersStructure);
    rawEyePositionTraces = [];
    usefulEyePositionTraces = [];
    timeArray = [];
    statisticsStructure = struct();
    return;
else
    RevasWarning(['StripAnalysis() is proceeding and overwriting an existing file. (' rawOutputFileName ')'], parametersStructure);  
end

if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['StripAnalysis() did not execute because it would overwrite existing file. (' outputFileName ')'], parametersStructure);
    rawEyePositionTraces = [];
    usefulEyePositionTraces = [];
    timeArray = [];
    statisticsStructure = struct();
    return;
else
    RevasWarning(['StripAnalysis() is proceeding and overwriting an existing file. (' outputFileName ')'], parametersStructure);  
end

%% Preallocation and variable setup
[stripIndices, stripsPerFrame] = DivideIntoStrips(videoInput, videoFrameRate, parametersStructure);
numberOfStrips = size(stripIndices, 1);

% two columns for horizontal and vertical movements
rawEyePositionTraces = NaN(numberOfStrips, 2);

% arrays for peak and second highest peak values
peakValueArray = zeros(numberOfStrips, 1);
secondPeakValueArray = zeros(numberOfStrips, 1);

% array for standard deviations (used for gaussian peaks approach)
standardDeviationsArray = NaN(numberOfStrips, 2);

% array for search windows
estimatedStripYLocations = NaN(numberOfStrips, 1);
searchWindowsArray = NaN(numberOfStrips, 2);

%% Populate time array
timeArray = (1:numberOfStrips)' / samplingRate;

%% GPU Preparation
% *** TODO: need GPU device to confirm ***
% Check if a GPU device is connected. If so, run calculations on the GPU
% (if enabled by the user).
enableGPU = (gpuDeviceCount > 0) & parametersStructure.enableGPU;
if enableGPU
    referenceFrame = gpuArray(referenceFrame);
end

%% Adaptive Search
% Estimate peak locations if adaptive search is enabled

if adaptiveSearch
    % Scale down the reference frame to a smaller size
    scaledDownReferenceFrame = referenceFrame( ...
        1:scalingFactor:end, ...
        1:scalingFactor:end);

    for frameNumber = (1:size(videoInput, 3))
        frame = videoInput(:,:,frameNumber);

        % Scale down the current frame to a smaller size as well
        scaledDownFrame = frame( ...
            1:scalingFactor:end, ...
            1:scalingFactor:end);

        correlationMap = normxcorr2(scaledDownFrame, scaledDownReferenceFrame);

        [~, yPeak, ~, ~] = ...
            FindPeak(correlationMap, parametersStructure);

        % Account for padding introduced by normxcorr2
        yPeak = yPeak - (size(scaledDownFrame, 1) - 1);

        % Populate search windows array but only fill in coordinates for the
        % top strip of each frame
        estimatedStripYLocations((frameNumber - 1) * stripsPerFrame + 1,:) = yPeak;
    end

    % Finish populating search window by taking the line between the top left
    % corner of the previous frame and the bottom left corner of the current
    % frame and dividing that line up by the number of strips per frame.
    for frameNumber = (1:size(videoInput, 3)-1)
        previousFrameYCoordinate = ...
            estimatedStripYLocations((frameNumber - 1) * stripsPerFrame + 1);
        currentFrameYCoordinate = ...
            estimatedStripYLocations((frameNumber) * stripsPerFrame + 1)...
            + size(scaledDownFrame, 1);

        % change per strip is determined by drawing a line from the top left
        % corner of the previous frame and the bottom left corner of the
        % current frame and then dividing it by the number of strips. Each time
        % we add change per strip, we thus take a step closer to the latter
        % point from the previous point and will arrive there after taking the
        % same number of steps as we have strips per frame.
        changePerStrip = (currentFrameYCoordinate - previousFrameYCoordinate) ...
            / stripsPerFrame;

        % For each strip, take the previous strip's value and add the change
        % per strip.
        for stripNumber = (2:stripsPerFrame)
            estimatedStripYLocations((frameNumber - 1) * stripsPerFrame + stripNumber) ...
                = estimatedStripYLocations((frameNumber - 1) * stripsPerFrame + stripNumber - 1) ...
                + changePerStrip;
        end
    end

    % Scale back up
    estimatedStripYLocations = (estimatedStripYLocations - 1) ...
        * scalingFactor + 1;

end

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Call normxcorr2() on each strip
% Note that calculation for each array value does not end with this loop,
% the logic below the loop in this section perform remaining operations on
% the values but are done outside of the loop in order to take advantage of
% vectorization (that is, if verbosity is not enabled since if it was, then
% these operations must be computed immediately so that the correct eye
% trace values can be plotted as early as possible).
for stripNumber = (1:numberOfStrips)
    
    if ~abortTriggered
        
        gpuTask = getCurrentTask;

        % Note that only one core should use the GPU at a time.
        % i.e. when processing multiple videos in parallel, only one should
        % use GPU.
        if enableGPU
            stripData = gpuArray(stripIndices(stripNumber,:));
        else
            stripData = stripIndices(stripNumber,:);
        end

        frame = stripData(1,3);

        if ismember(frame, badFrames)
            rawEyePositionTraces(stripNumber,:) = [NaN NaN];
            peakValueArray(stripNumber) = NaN;
            secondPeakValueArray(stripNumber) = NaN;
            continue;
        end

        rowStart = stripData(1,1);
        columnStart = stripData(1,2);
        rowEnd = rowStart + stripHeight - 1;
        columnEnd = columnStart + stripWidth - 1;
        strip = videoInput(rowStart:rowEnd, columnStart:columnEnd, frame);
        
        correlationMap = normxcorr2(strip, referenceFrame);
        parametersStructure.stripNumber = stripNumber;  
        parametersStructure.stripsPerFrame = stripsPerFrame;

        if adaptiveSearch ...
                && ~isnan(estimatedStripYLocations(stripNumber))
            % Cut out a smaller search window from correlation.
            upperBound = floor(min(max(1, ...
                estimatedStripYLocations(stripNumber) ...
                - ((searchWindowHeight - stripHeight)/2)), ...
                size(videoInput, 1)));
            lowerBound = floor(min(size(videoInput, 1), ...
                estimatedStripYLocations(stripNumber) ...
                + ((searchWindowHeight - stripHeight)/2) ...
                + stripHeight));
            adaptedCorrelation = correlationMap(upperBound:lowerBound,1:end);
            
            try
                % Try to use adapted version of correlation map.
                [xPeak, yPeak, peakValue, secondPeakValue] = ...
                    FindPeak(adaptedCorrelation, parametersStructure);
  
                % See if adapted result is acceptable or not.
                if ~enableGaussianFiltering && ...
                        (peakValue <= 0 || secondPeakValue <= 0 ...
                        || secondPeakValue / peakValue > maximumPeakRatio ...
                        || peakValue < minimumPeakThreshold)
                    % Not acceptable, try again in the catch block with full correlation map.
                    error('Jumping to catch block immediately below.');
                elseif enableGaussianFiltering % TODO, need to test.
                    % Middle row SDs in column 1, Middle column SDs in column 2.
                    middleRow = ...
                        adaptedCorrelation(max(ceil(yPeak-...
                            SDWindowSize/2/scalingFactor),...
                            1): ...
                        min(floor(yPeak+...
                            SDWindowSize/2/scalingFactor),...
                            size(adaptedCorrelation,1)), ...
                            floor(xPeak));
                    middleCol = ...
                        adaptedCorrelation(floor(yPeak), ...
                        max(ceil(xPeak-...
                            SDWindowSize/2/scalingFactor),...
                            1): ...
                        min(floor(xPeak+SDWindowSize/2/scalingFactor),...
                            size(adaptedCorrelation,2)))';
                    fitOutput = fit(((1:size(middleRow,1))-ceil(size(middleRow,1)/2))', middleRow, 'gauss1');
                    isAcceptable = true;
                    if fitOutput.c1 > maximumSD
                        isAcceptable = false;
                    end
                    fitOutput = fit(((1:size(middleCol,1))-ceil(size(middleCol,1)/2))', middleCol, 'gauss1');
                    if fitOutput.c1 > maximumSD
                        isAcceptable = false;
                    end
                    clear fitOutput;
                    if peakValue <= 0 ...
                        || ~isAcceptable ...
                        || peakValue < minimumPeakThreshold
                        % Not acceptable, try again in the catch block with full correlation map.
                        error('Jumping to catch block immediately below.'); 
                    end
                end
                correlationMap = adaptedCorrelation;
                searchWindowsArray(stripNumber,:) = [upperBound lowerBound];
            catch
                upperBound = 1;
                % It failed or was unacceptable, so use full correlation map.
                [xPeak, yPeak, peakValue, secondPeakValue] = ...
                    FindPeak(correlationMap, parametersStructure);
    
                searchWindowsArray(stripNumber,:) = [NaN NaN];
            end
        else
            upperBound = 1;
            [xPeak, yPeak, peakValue, secondPeakValue] = ...
                FindPeak(correlationMap, parametersStructure);        
        end

        % 2D Interpolation if enabled
        if enableSubpixelInterpolation
            [interpolatedPeakCoordinates, statisticsStructure.errorStructure] = ...
                Interpolation2D(correlationMap, [yPeak, xPeak], ...
                subpixelInterpolationParameters);

            xPeak = interpolatedPeakCoordinates(2);
            yPeak = interpolatedPeakCoordinates(1);      
        end

        % If GPU was used, transfer peak values and peak locations
        if enableGPU
            xPeak = gather(xPeak, gpuTask.ID);
            yPeak = gather(yPeak, gpuTask.ID);
            peakValue = gather(peakValue, gpuTask.ID);
            secondPeakValue = gather(secondPeakValue, gpuTask.ID);
        end

        if enableGaussianFiltering
            % Fit a gaussian in a pixel window around the identified peak.
            % The pixel window is of size
            % |SDWindowSize| x
            % |SDWindowSize/2|
            %
            % Take the middle row and the middle column, and fit a one-dimensional
            % gaussian to both in order to get the standard deviations.
            % Store results in statisticsStructure for choosing bad frames
            % later.

            % Middle row SDs in column 1, Middle column SDs in column 2.
            middleRow = ...
                correlationMap(max(ceil(yPeak-SDWindowSize/2), 1): ...
                min(floor(yPeak+SDWindowSize/2), size(correlationMap,1)), ...
                floor(xPeak));
            middleCol = ...
                correlationMap(floor(yPeak), ...
                max(ceil(xPeak-SDWindowSize/2), 1): ...
                min(floor(xPeak+SDWindowSize/2), size(correlationMap,2)))';
            fitOutput = fit(((1:size(middleRow,1))-ceil(size(middleRow,1)/2))', middleRow, 'gauss1');
            standardDeviationsArray(stripNumber, 1) = fitOutput.c1;
            fitOutput = fit(((1:size(middleCol,1))-ceil(size(middleCol,1)/2))', middleCol, 'gauss1');
            standardDeviationsArray(stripNumber, 2) = fitOutput.c1;
            clear fitOutput;
        end

        % Show surface plot for this correlation if verbosity enabled
        if enableVerbosity
            if enableGPU
                correlationMap = gather(correlationMap, gpuTask.ID);
            end
            if isfield(parametersStructure, 'axesHandles')
                axes(parametersStructure.axesHandles(1));
                colormap(parametersStructure.axesHandles(1), 'default');
            else
                figure(1);
            end
            [surfX,surfY] = meshgrid(1:size(correlationMap,2), 1:size(correlationMap,1));
            surf(surfX, surfY, correlationMap, 'linestyle', 'none');
            title([num2str(stripNumber) ' out of ' num2str(numberOfStrips)]);
            xlim([1 size(correlationMap,2)]);
            ylim([1 size(correlationMap,1)]);
            zlim([-1 1]);

            % Mark the identified peak on the plot with an arrow.
            text(xPeak, yPeak, peakValue, '\downarrow', 'Color', 'red', ...
                'FontSize', 20, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

            drawnow;  
        end
 
        % If these peaks are in terms of an adapted correlation map, restore it
        % back to in terms of the full map.
        yPeak = yPeak + upperBound - 1;
        rawEyePositionTraces(stripNumber,:) = [xPeak yPeak];
        
        peakValueArray(stripNumber) = peakValue;
        secondPeakValueArray(stripNumber) = secondPeakValue;

        % If verbosity is enabled, also show eye trace plot with points
        % being plotted as they become available.
        if enableVerbosity

            % Adjust for padding offsets added by normxcorr2()
            % If we enable verbosity and demand that we plot the points as we
            % go, then adjustments must be made here in order for the plot to
            % be interpretable.
            % Therefore, we will only perform these same operations after the
            % loop to take advantage of vectorization only if they are not
            % performed here, namely, if verbosity is not enabled and this
            % if statement does not execute.
            rawEyePositionTraces(stripNumber,2) = ...
                rawEyePositionTraces(stripNumber,2) - (stripHeight - 1);
            rawEyePositionTraces(stripNumber,1) = ...
                rawEyePositionTraces(stripNumber,1) - (stripWidth - 1);

            % We must subtract back out the expected strip coordinates in order
            % to obtain the net movement (the net difference between no
            % movement and the movement that was observed).
            rawEyePositionTraces(stripNumber,1) = ...
                rawEyePositionTraces(stripNumber,1) - stripIndices(stripNumber,2);
            rawEyePositionTraces(stripNumber,2) = ...
                rawEyePositionTraces(stripNumber,2) - stripIndices(stripNumber,1);

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

%% Adjust for padding offsets added by normxcorr2()
% Do this after the loop to take advantage of vectorization
% Only run this section if verbosity was not enabled. If verbosity was
% enabled, then these operations were already performed for each point
% before it was plotted to the eye traces graph. If verbosity was not
% enabled, then we do it now in order to take advantage of vectorization.
if ~parametersStructure.enableVerbosity
    rawEyePositionTraces(:,2) = ...
        rawEyePositionTraces(:,2) - (stripHeight - 1);
    rawEyePositionTraces(:,1) = ...
        rawEyePositionTraces(:,1) - (stripWidth - 1);

    % We must subtract back out the starting coordinates in order
    % to obtain the net movement (comparing expected strip locations if
    % there were no movement to observed location).
    rawEyePositionTraces(:,1) = rawEyePositionTraces(:,1) - stripIndices(:,2);
    rawEyePositionTraces(:,2) = rawEyePositionTraces(:,2) - stripIndices(:,1);

    % Negate eye position traces to flip directions.
    rawEyePositionTraces = -rawEyePositionTraces;
end

%% Populate statisticsStructure

statisticsStructure.peakValues = peakValueArray;
statisticsStructure.peakRatios = secondPeakValueArray ./ peakValueArray;
statisticsStructure.searchWindows = searchWindowsArray;
statisticsStructure.standardDeviations = standardDeviationsArray;

%% Populate usefulEyePositionTraces

if enableGaussianFiltering
    % Criteria for gaussian filtering method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * after a guassian is fitted in a 25x25 pixel window around the
    %   identified peak, the standard deviation of the curve must be below
    %   a maximum threshold in both the horizontal and vertical axes.
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToKeep = (statisticsStructure.standardDeviations(:,1) <= maximumSD)...
        & (statisticsStructure.standardDeviations(:,2) <= maximumSD)...
        & (statisticsStructure.peakValues >= minimumPeakThreshold);
else
    % Criteria for peak ratio method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * the ratio of the second peak to the first peak is smaller than a
    %   maximum threshold (they must be sufficiently distinct).
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToKeep = (statisticsStructure.peakRatios <= maximumPeakRatio)...
        & (statisticsStructure.peakValues >= minimumPeakThreshold);
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
end

%% Save to output mat file

if ~abortTriggered && ~isempty(inputVideoPath)
    % Save a copy of the raw traces.
    eyePositionTraces = rawEyePositionTraces;
    save(rawOutputFileName, 'eyePositionTraces', 'timeArray', ...
        'parametersStructure', 'referenceFramePath');
    
    % Save a copy of the useful traces (under file labeled 'final').
    eyePositionTraces = usefulEyePositionTraces;
    save(outputFileName, 'eyePositionTraces', 'timeArray', ...
        'parametersStructure', 'referenceFramePath');
end
end
