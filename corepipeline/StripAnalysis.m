function [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure]...
    = StripAnalysis(videoInput, referenceFrame, parametersStructure)
%STRIP ANALYSIS Extract eye movements in units of pixels.
%   Cross-correlation of horizontal strips with a pre-defined
%   reference frame.

%% Input Validation

inputVideoPath = '';
referenceFramePath = '';

% If videoInput is a character array, then a path was passed in.
% Attempt to convert it to a 3D or 4D array, depending on number of
% color channels.
if ischar(videoInput)
    inputVideoPath = videoInput;
    [videoInput, videoFrameRate] = VideoPathToArray(videoInput);
else
    % ASSUMPTION
    % If only a raw matrix is provided, then we will take the frame rate to
    % be 30.
    % TODO
    warning('A raw matrix was provided; assuming that frame rate is 30 fps.');
    videoFrameRate = 30;
end

% If referenceFrame is a character array, then a path was passed in.
if ischar(referenceFrame)
    referenceFramePath = referenceFrame;
    referenceFrame = importdata(referenceFramePath);
end

% Set strip width if not provided
if ~isfield(parametersStructure, 'stripWidth')
    stripWidth = size(videoInput, 2);
else
    stripWidth = parametersStructure.stripWidth;
end

ValidateVideoInput(videoInput);
ValidateReferenceFrame(referenceFrame);
ValidateParametersStructure(parametersStructure);

% Identify which frames are bad frames
% The filename may not exist if a raw array was passed in.
if ~isfield(parametersStructure, 'badFrames')
    nameEnd = inputVideoPath(1:size(inputVideoPath, 2)-4);
    blinkFramesPath = [nameEnd '_blinkframes.mat'];
    try
        load(blinkFramesPath, 'badFrames');
        parametersStructure.badFrames = badFrames;
    catch
        parametersStructure.badFrames = [];
    end
end

% *** TODO: needs testing on color video ***
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

%% Handle overwrite scenarios.

outputFileName = [inputVideoPath(1:end-4) '_' ...
    int2str(parametersStructure.samplingRate) '_hz_final'];

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
timeArray = (1:numberOfStrips)' / parametersStructure.samplingRate;

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

if parametersStructure.adaptiveSearch
    % Scale down the reference frame to a smaller size
    scaledDownReferenceFrame = referenceFrame( ...
        1:parametersStructure.adaptiveSearchScalingFactor:end, ...
        1:parametersStructure.adaptiveSearchScalingFactor:end);

    for frameNumber = (1:size(videoInput, 3))
        frame = videoInput(:,:,frameNumber);

        % Scale down the current frame to a smaller size as well
        scaledDownFrame = frame( ...
            1:parametersStructure.adaptiveSearchScalingFactor:end, ...
            1:parametersStructure.adaptiveSearchScalingFactor:end);

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
        * parametersStructure.adaptiveSearchScalingFactor + 1;

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
            localParametersStructure = gpuArray(parametersStructure);
            stripData = gpuArray(stripIndices(stripNumber,:));
        else
            localParametersStructure = parametersStructure;
            stripData = stripIndices(stripNumber,:);
        end

        frame = stripData(1,3);

        if ismember(frame, localParametersStructure.badFrames)
            rawEyePositionTraces(stripNumber,:) = [NaN NaN];
            peakValueArray(stripNumber) = NaN;
            secondPeakValueArray(stripNumber) = NaN;
            continue
        end

        rowStart = stripData(1,1);
        columnStart = stripData(1,2);
        rowEnd = rowStart + localParametersStructure.stripHeight - 1;
        columnEnd = columnStart + stripWidth - 1;
        strip = videoInput(rowStart:rowEnd, columnStart:columnEnd, frame);
        
        correlationMap = normxcorr2(strip, referenceFrame);
        parametersStructure.stripNumber = stripNumber;  
        parametersStructure.stripsPerFrame = stripsPerFrame;

        if parametersStructure.adaptiveSearch ...
                && ~isnan(estimatedStripYLocations(stripNumber))
            % Cut out a smaller search window from correlation.
            upperBound = floor(min(max(1, ...
                estimatedStripYLocations(stripNumber) ...
                - ((parametersStructure.searchWindowHeight - parametersStructure.stripHeight)/2)), ...
                size(videoInput, 1)));
            lowerBound = floor(min(size(videoInput, 1), ...
                estimatedStripYLocations(stripNumber) ...
                + ((parametersStructure.searchWindowHeight - parametersStructure.stripHeight)/2) ...
                + parametersStructure.stripHeight));
            adaptedCorrelation = correlationMap(upperBound:lowerBound,1:end);
            
            try
                % Try to use adapted version of correlation map.
                [xPeak, yPeak, peakValue, secondPeakValue] = ...
                    FindPeak(adaptedCorrelation, parametersStructure);
  
                % See if adapted result is acceptable or not.
                if ~parametersStructure.enableGaussianFiltering && ...
                        (peakValue <= 0 || secondPeakValue <= 0 ...
                        || secondPeakValue / peakValue > parametersStructure.maximumPeakRatio ...
                        || peakValue < parametersStructure.minimumPeakThreshold)
                    % Not acceptable, try again in the catch block with full correlation map.
                    error('Jumping to catch block immediately below.');
                end
                correlationMap = adoptedCorrelation;
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
        if localParametersStructure.enableSubpixelInterpolation
            [interpolatedPeakCoordinates, statisticsStructure.errorStructure] = ...
                Interpolation2D(correlationMap, [yPeak, xPeak], ...
                localParametersStructure.subpixelInterpolationParameters);

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

        if parametersStructure.enableGaussianFiltering
            % Fit a gaussian in a pixel window around the identified peak.
            % The pixel window is of size
            % |parametersStructure.SDWindowSize| x
            % |parametersStructure.SDWindowSize/2|
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

        % Show surface plot for this correlation if verbosity enabled
        if localParametersStructure.enableVerbosity
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
        if localParametersStructure.enableVerbosity

            % Adjust for padding offsets added by normxcorr2()
            % If we enable verbosity and demand that we plot the points as we
            % go, then adjustments must be made here in order for the plot to
            % be interpretable.
            % Therefore, we will only perform these same operations after the
            % loop to take advantage of vectorization only if they are not
            % performed here, namely, if verbosity is not enabled and this
            % if statement does not execute.
            rawEyePositionTraces(stripNumber,2) = ...
                rawEyePositionTraces(stripNumber,2) - (parametersStructure.stripHeight - 1);
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
        rawEyePositionTraces(:,2) - (parametersStructure.stripHeight - 1);
    rawEyePositionTraces(:,1) = ...
        rawEyePositionTraces(:,1) - (stripWidth - 1);

    % Adjust in vertical direction.
    % We must subtract back out the starting strip vertical coordinate in order
    % to obtain the net vertical movement.
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

if parametersStructure.enableGaussianFiltering
    % Criteria for gaussian filtering method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * after a guassian is fitted in a 25x25 pixel window around the
    %   identified peak, the standard deviation of the curve must be below
    %   a maximum threshold in both the horizontal and vertical axes.
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToRemove = (statisticsStructure.standardDeviations(:,1) <= parametersStructure.maximumSD)...
        & (statisticsStructure.standardDeviations(:,2) <= parametersStructure.maximumSD)...
        & (statisticsStructure.peakValues >= parametersStructure.minimumPeakThreshold);
else
    % Criteria for peak ratio method is to ensure that:
    %   * the peak value is above a minimum threshold
    %   * the ratio of the second peak to the first peak is smaller than a
    %   maximum threshold (they must be sufficiently distinct).
    
    % Determine which eye traces to throw out
    % 1 = keep, 0 = toss
    eyeTracesToRemove = (statisticsStructure.peakRatios <= parametersStructure.maximumPeakRatio)...
        & (statisticsStructure.peakValues >= parametersStructure.minimumPeakThreshold);
end

% convert logical array to double array
eyeTracesToRemove = double(eyeTracesToRemove);

% change all 0 = toss to be NaN = toss
eyeTracesToRemove(eyeTracesToRemove == 0) = NaN;

% multiply each component by 1 to keep eyePositionTraces or by NaN to toss.
eyeTracesToRemove = repmat(eyeTracesToRemove,1,2); % duplicate vector first
usefulEyePositionTraces = rawEyePositionTraces .* eyeTracesToRemove;

%% Plot Useful Eye Traces
if ~abortTriggered && parametersStructure.enableVerbosity
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(3));
        colormap(parametersStructure.axesHandles(3), 'gray');
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

%% Save to output mat file

if ~abortTriggered && ~isempty(inputVideoPath)
    eyePositionTraces = usefulEyePositionTraces;

    save(outputFileName, 'eyePositionTraces', 'timeArray', ...
        'parametersStructure', 'referenceFramePath');
end
end