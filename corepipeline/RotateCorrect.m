function [coarseRefFrame, coordinatesAndDegrees] = RotateCorrect(shrunkFrames, bigFrames, ...
    referenceFrame, outputFileName, parametersStructure)
%%ROTATE CORRECT     RotateCorrect takes in the 3D video array from
%                    CoarseRef and checks each frame to see whether
%                    rotation is necessary (using peak ratios as the
%                    criterion). If the frame's peak ratio is above a
%                    specified threshold, then the function rotates the
%                    frame and cross-correlates until the ratio dips below 
%                    the threshold. It returns a coarseRefFrame using this
%                    correction method, as well as a matrix containing
%                    information regarding which degree rotation was used,
%                    and what each frame's peak ratio was.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  degreeRange         :   specifies the degree range to test for each 
%                          frame, passed in as a scalar (i.e., passing in 5
%                          means that the function will test rotations from
%                          -5 to +5 degrees).
%  peakDropWindow      :   size of the correlation map to exclude when
%                          finding the second peak, for calculating peak 
%                          ratios. Units are in pixels. 
%  scalingFactor       :   the amount by which each frame is scaled down
%                          for the coarse cross-correlating
%                          0 < scalingFactor <= 1
%  rotateMaximumPeakRatio : specifies the peak ratio threshold below which 
%                           a rotated frame will be considered a "good" 
%                           frame. 
%
%   Example usage: 
%       
%   Example usage: 
%       videoPath = 'MyVid.avi';
%       load('MyVid_params.mat')
%       coarseParameters.degreeRange = 1;
%       coarseParameters.peakDropWindow = 25;
%       coarseParameters.rotateMaximumPeakRatio = 0.6;
%       coarseReferenceFrame = RotateCorrect(filename, coarseParameters);

%% Initialize variables
if ~isfield(parametersStructure, 'degreeRange')
    parametersStructure.degreeRange = 5;
end

% Preallocate the rotate corrected coarse reference frame and counter array
rotateCorrectedCoarse = zeros(size(bigFrames, 1)*2.5, size(bigFrames, 2)*2.5);
counterArray = rotateCorrectedCoarse;

% usefulEyePositionTraces contains: Columns 1 & 2 (coordinates) and Column
% 3 (degree that returns the ideal rotation)
rotations = -parametersStructure.degreeRange:0.1:parametersStructure.degreeRange;
coordinatesAndDegrees = zeros(size(shrunkFrames, 3), 3);

% Each time StripAnalysis is called, we must pass in two frames in a 3D
% matrix. To save time, skip the second frame, so just mark it as a bad
% frame. Only the first frame has the frame of interest
parametersStructure.badFrames = 2;

%% Examine each frame
for frameNumber = 1:size(shrunkFrames, 3)
    
    frame = shrunkFrames(:, :, frameNumber);
    
    % First check if the frame has a peak ratio lower than the
    % designated threshold. If so, then update the reference frame
    % and move on to the next frame. Because StripAnalysis only accepts 3D
    % matricies, convert frame into 3D matrix first. Skip this first
    % iteration if this function call came from reRef because the only
    % reason reReference would call this function is if the peakRatio was
    % not sufficient (i.e., we don't need to check 0 degrees anymore).
    
    threeDimensionalFrame = zeros(size(frame, 1), size(frame, 2), 2);
    threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + ...
        double(frame);
    
    [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
        threeDimensionalFrame, referenceFrame, parametersStructure);
    
    peakRatio = statisticsStructure.peakRatios(1);
    usefulEyePositionTraces = usefulEyePositionTraces(1, :);
    
    if peakRatio <= parametersStructure.rotateMaximumPeakRatio && ismember(1, ...
            ~isnan(usefulEyePositionTraces))
        
        coordinatesAndDegrees(frameNumber, 1:2) = usefulEyePositionTraces ...
            * 1/parametersStructure.scalingFactor;
        coordinatesAndDegrees(frameNumber, 3) = 0;
        
        coordinates = coordinatesAndDegrees(frameNumber, 1:2);
        coordinates = round(ScaleCoordinates(coordinates));
        
        minRow = coordinates(1, 2);
        minColumn = coordinates(1, 1);
        maxRow = size(bigFrames, 1) + minRow - 1;
        maxColumn = size(bigFrames, 2) + minColumn - 1;
        
        selectRow = round(minRow):round(maxRow);
        selectColumn = round(minColumn):round(maxColumn);
        
        bigFrame = double(bigFrames(:, :, frameNumber));
        
        rotateCorrectedCoarse(selectRow, selectColumn) = ...
            rotateCorrectedCoarse(selectRow, selectColumn) + bigFrame;
        counterArray(selectRow, selectColumn) = counterArray(selectRow, ...
            selectColumn) + 1;
        
        % Now update the "temporary" reference frame for future frames
        referenceFrame = rotateCorrectedCoarse./counterArray;
        referenceFrame = Crop(referenceFrame);
        referenceFrame = imresize(referenceFrame, parametersStructure.scalingFactor);
        continue
    end
    
    % If the correlation value does not pass the threshold, start
    % rotating frames.
    
    for k = 1:max(size(rotations))
        
        rotation = rotations(k);
        
        % Already checked whether the frame without rotation exceeds
        % the threshold, so skip this iteration to reduce runtime
        if rotation == 0
            continue
        end
        
        tempFrame = imrotate(frame, rotation);
        
        threeDimensionalFrame = zeros(size(tempFrame, 1), size(tempFrame, 2), 2);
        threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + ...
            double(tempFrame);
        
        [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
            threeDimensionalFrame, referenceFrame, parametersStructure);
        
        peakRatio = statisticsStructure.peakRatios(1);
        usefulEyePositionTraces = usefulEyePositionTraces(1, :);
        
        if isnan(usefulEyePositionTraces)
            continue
        end
        
        % If the peakRatio for this rotation is lower than the threshold,
        % then move on to the next frame by breaking out of the
        % rotations for-loop. Otherwise, continue rotating
        if peakRatio <= parametersStructure.rotateMaximumPeakRatio
            
            coordinatesAndDegrees(frameNumber, 1:2) = ...
                usefulEyePositionTraces * 1/parametersStructure.scalingFactor;
            coordinatesAndDegrees(frameNumber, 3) = rotation;
            
            coordinates = coordinatesAndDegrees(frameNumber, 1:2);
            coordinates = round(ScaleCoordinates(coordinates));
            
            minRow = coordinates(1, 2);
            minColumn = coordinates(1, 1);
            maxRow = size(bigFrames, 1) + minRow - 1;
            maxColumn = size(bigFrames, 2) + minColumn - 1;
            
            bigFrame = bigFrames(:, :, frameNumber);
            bigFrame = double(imrotate(bigFrame, rotation));
            
            % Because rotating changes the dimensions of the image
            % slightly, extend selectRow and selectColumn
            selectRow = round(minRow):round(maxRow);
            selectColumn = round(minColumn):round(maxColumn);
            if max(size(selectRow)) < size(bigFrame, 1)
                difference = size(bigFrame, 1) - max(size(selectRow));
                lastNumber = selectRow(end);
                selectRow(end+1:end+difference) = lastNumber+1:...
                    lastNumber + difference;
            end
            
            if max(size(selectColumn)) < size(bigFrame, 2)
                difference = size(bigFrame, 2) - max(size(selectColumn));
                lastNumber = selectColumn(end);
                selectColumn(end+1:end+difference) = lastNumber+1:...
                    lastNumber + difference;
            end
            
            rotateCorrectedCoarse(selectRow, selectColumn) = ...
                rotateCorrectedCoarse(selectRow, selectColumn) + bigFrame;
            counterArray(selectRow, selectColumn) = counterArray(selectRow, ...
                selectColumn) + 1;
            
            % Now update the "temporary" reference frame for future frames
            referenceFrame = rotateCorrectedCoarse./counterArray;
            referenceFrame = Crop(referenceFrame);
            referenceFrame = imresize(referenceFrame, parametersStructure.scalingFactor);
            break
        else
            continue
        end
        
    end
end

coarseRefFrame = rotateCorrectedCoarse./counterArray;
coarseRefFrame = Crop(coarseRefFrame);
coarseRefFrame = double(coarseRefFrame)/255;
save(outputFileName, 'coarseRefFrame');

if parametersStructure.enableVerbosity >= 1
    if isfield(parametersStructure, 'axesHandles')
        axes(parametersStructure.axesHandles(3));
        colormap(parametersStructure.axesHandles(3), 'gray');
    else
        figure('Name', 'Coarse Reference Frame');
    end
    imshow(coarseRefFrame);
end
end
