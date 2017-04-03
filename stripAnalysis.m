function [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure]...
    = stripAnalysis(videoInput, referenceFrame, parametersStructure)
%STRIP ANALYSIS Extract eye movements in units of pixels.
%   Cross-correlation of horizontal strips with a pre-defined
%   reference frame. 

%% Input Validation

% If videoInput is a character array, then a path was passed in.
% Attempt to convert it to a 3D or 4D array, depending on number of
% color channels.
if ischar(videoInput)
    [videoInput, videoFrameRate] = videoPathToArray(videoInput);
else
    % ASSUMPTION
    % If only a raw matrix is provided, then we will take the frame rate to
    % be 30.
    % TODO
    warning('A raw matrix was provided; assuming that frame rate is 30 fps.');
    videoFrameRate = 30;
end

validateVideoInput(videoInput);
validateReferenceFrame(referenceFrame);
validateParametersStructure(parametersStructure);

% UNTESTED, needs testing on color video
% Change 4D arrays to 3D by making video grayscale
if ndims(videoInput) == 4
    numberOfFrames = size(videoInput, 3);
    newVideoInput = squeeze(videoInput(:,:,:,1));
    for frame = (1:numberOfFrames)
        frame3D = squeeze(videoInput(:,:,frame,:));
        newVideoInput(:,:,frame) = rgb2gray(frame3D);
    end
    videoInput = newVideoInput;
end

%% normxcorr2() on each strip

stripIndices = divideIntoStrips(videoInput, videoFrameRate, parametersStructure);

% Preallocate output and helper matrices
numberOfStrips = size(stripIndices, 1);
% two columns for horizontal and vertical movements
rawEyePositionTraces = zeros(numberOfStrips, 2);
% only one column for timeArray
timeArray = zeros(numberOfStrips, 1);
% arrays for peak and second highest peak values
peakValues = zeros(numberOfStrips, 1);
secondHighestPeakValues = zeros(numberOfStrips, 1);

% We set the number of workers for the parfor loop below here.
% If the verbosity flag in the parameters is disabled, then we use a normal
% parfor loop with whatever number of workers is available. If the
% verbosity flag is enabled, then we must run the loop in sequence in order
% for the surface plots to be displayed in order (essentially by using 0
% workers for our parfor loop).
numberOfWorkers = numlabs * double(~parametersStructure.enableVerbosity);

% Note that calculation for each array value does not end with this loop,
% the logic below the loop in this section perform remaining operations on
% the values but are done outside of the loop in order to take advantage of
% vectorization.

% Two for loop headers here. Use regular for loop for debugging, use
% parfor version for optimal execution.
parfor (stripNumber = (1:numberOfStrips), numberOfWorkers)
%for stripNumber = (1:numberOfStrips)
    localParametersStructure = parametersStructure;
    stripData = stripIndices(stripNumber,:);
    rowStart = stripData(1,1);
    columnStart = stripData(1,2);
    frame = stripData(1,3);
    if ismember(frame, localParametersStructure.badFrames)
        continue
    end
    rowEnd = rowStart + localParametersStructure.stripHeight - 1;
    columnEnd = columnStart + localParametersStructure.stripWidth - 1;
    
    strip = videoInput(rowStart:rowEnd, columnStart:columnEnd, frame);
    
    % correlation = normxcorr2(videoInput(...
    %     rowStart:rowEnd, columnStart:columnEnd, frame),referenceFrame);
    correlation = normxcorr2(strip, referenceFrame);
    
    % Show surface plot for this correlation if verbosity enabled
    if localParametersStructure.enableVerbosity
        figure(1);
        [surfX,surfY] = meshgrid(1:size(correlation,2), 1:size(correlation,1));
        surf(surfX, surfY, correlation,'linestyle','none');
        title([num2str(stripNumber) ' out of ' num2str(numberOfStrips)]);
        xlim([1 size(correlation,2)]);
        ylim([1 size(correlation,1)]);
        zlim([-1 1]);
        drawnow;
    end
    
    % Make copy of correlation map to find peaks
    correlationCopy = correlation;
    
    % Find peak of correlation map
    [ypeak, xpeak] = find(correlationCopy==max(correlationCopy(:)));
    peakValues(stripNumber) = max(correlationCopy(:));
    
    % Find second highest point of correlation map
    correlationCopy(ypeak, xpeak) = -inf;
    secondHighestPeakValues(stripNumber) = max(correlationCopy(:));
    
    % 2D Interpolation if enabled
    if localParametersStructure.enableSubpixelInterpolation
        localParametersStructure.subpixelInterpolationParameters.enableVerbosity = ...
            localParametersStructure.enableVerbosity;
        
        [interpolatedPeakCoordinates, errorStructure] = ...
            interpolation2D(correlation, [xpeak, ypeak], ...
            localParametersStructure.subpixelInterpolationParameters);
        
        xpeak = interpolatedPeakCoordinates(1);
        ypeak = interpolatedPeakCoordinates(2);      
    end

    rawEyePositionTraces(stripNumber,:) = [xpeak ypeak];
end

% Adjust for padding offsets added by normxcorr2()
% Do this after the loop to take advantage of vectorization
rawEyePositionTraces(:,2) = rawEyePositionTraces(:,2) - (parametersStructure.stripHeight - 1);
rawEyePositionTraces(:,1) = rawEyePositionTraces(:,1) - (parametersStructure.stripWidth - 1);

% Adjust in vertical direction.
% We must subtract back out the starting strip vertical coordinate in order
% to obtain the net vertical movement.
rawEyePositionTraces(:,1) = rawEyePositionTraces(:,1) - stripIndices(:,2);
rawEyePositionTraces(:,2) = rawEyePositionTraces(:,2) - stripIndices(:,1);

% Negate eye position traces to flip directions.
rawEyePositionTraces = -rawEyePositionTraces;

%% Populate statisticsStructure

statisticsStructure.peakValues = peakValues;
statisticsStructure.peakRatios = peakValues ./ secondHighestPeakValues;
statisticsStructure.searchWindows = []; % TODO Unsure how to implement
statisticsStructure.errorStructure = struct();

%% Populate usefulEyePositionTraces

% Determine which eye traces to throw out
% 1 = keep, 0 = toss
eyeTracesToRemove = (statisticsStructure.peakRatios >= parametersStructure.minimumPeakRatio);

% convert logical array to double array
eyeTracesToRemove = double(eyeTracesToRemove);

% change all 0 = toss to be NaN = toss
eyeTracesToRemove(eyeTracesToRemove == 0) = NaN;

% multiply each component by 1 to keep eyePositionTraces or by NaN to toss.
eyeTracesToRemove = repmat(eyeTracesToRemove,1,2); % duplicate vector first
usefulEyePositionTraces = rawEyePositionTraces .* eyeTracesToRemove;

%% Populate timeArray

timeArray = (1:numberOfStrips)' / parametersStructure.samplingRate;

end