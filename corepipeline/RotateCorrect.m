function [coarseRefFrame, coordinatesAndDegrees] = RotateCorrect(shrunkFrames, bigFrames, ...
    referenceFrame, outputFileName, params)
%%ROTATE CORRECT     RotateCorrect takes in a frames and reference frame--
% both represented as 2D matrices--and returns a 4-column matrix. It
% rotates frames until they correlate with the reference frame with a 
% correlation value greater than or equal to the minimumPeakThreshold. If
% none of the rotations surpasses the threshold, then the function simply
% returns the optimal degree rotation for that frame.

%% Initialize variables
if ~isfield(params, 'degreeRange')
    params.degreeRange = 5;
end

% usefulEyePositionTraces contains: Columns 1 & 2 (coordinates) and Column
% 3 (degree that returns the ideal rotation)
rotations = -params.degreeRange:0.5:params.degreeRange;
coordinatesAndDegrees = zeros(size(shrunkFrames, 3), 3);

% Pre-allocate the template frame
if ~isfield(params, 'reRef')
    rotateCorrectedCoarse = zeros(size(bigFrames, 1)*2, size(bigFrames, 2)*2);
    counterArray = rotateCorrectedCoarse;
end

% Each time StripAnalysis is called, we must pass in two frames in a 3D
% matrix. To save time, skip the second frame, so just mark it as a bad
% frame. Only the first frame has the frame of interest
params.badFrames = 2;
%% Examine each frame
for frameNumber = 1:size(shrunkFrames, 3)
    % Just look at the first frame and stop if using this function for 
    % rereferencing
    if isfield(params, 'reRef') && frameNumber > 1
        break
    end
    frame = shrunkFrames(:, :, frameNumber);

    % First check if the frame has a peak ratio lower than the
    % designated threshold. If so, then update the reference frame 
    % and move on to the next frame. Because StripAnalysis only accepts 3D
    % matricies, convert frame into 3D matrix first. Skip this first
    % iteration if this function call came from reRef because the only
    % reason reReference would call this function is if the peakRatio was
    % not sufficient (i.e., we don't need to check 0 degrees anymore).
    
    if ~isfield(params, 'reRef')
        threeDimensionalFrame = zeros(size(frame, 1), size(frame, 2), 2);
        threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + ...
            double(frame);

        %params.peakDropWindow = 25;
        [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
            threeDimensionalFrame, referenceFrame, params);
    
        peakRatio = statisticsStructure.peakRatios(1);
        usefulEyePositionTraces = usefulEyePositionTraces(1, :);
    
        if peakRatio <= params.rotateMaximumPeakRatio && ismember(1, ...
                ~isnan(usefulEyePositionTraces))
            
            coordinatesAndDegrees(frameNumber, 1:2) = usefulEyePositionTraces ...
                * 1/params.scalingFactor;
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
            referenceFrame = imresize(referenceFrame, params.scalingFactor);
            continue
        end
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
        
%         %params.peakDropWindow = 25;
%         params.SDWindowSize = 25;
%         params.enableGaussianFiltering = true;
%         params.maximumSD = 25;
        [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
            threeDimensionalFrame, referenceFrame, params);
        
        peakRatio = statisticsStructure.peakRatios(1);
        usefulEyePositionTraces = usefulEyePositionTraces(1, :);
        
        if isnan(usefulEyePositionTraces)
            continue
        end
        
        disp(rotation)
        disp(peakRatio)
        disp(peakRatio <= params.rotateMaximumPeakRatio)
        
        % If the peakRatio for this rotation is lower than the threshold,
        % then move on to the next frame by breaking out of the
        % rotations for-loop. Otherwise, continue rotating
        if peakRatio <= params.rotateMaximumPeakRatio
            
            coordinatesAndDegrees(frameNumber, 1:2) = ...
                usefulEyePositionTraces * 1/params.scalingFactor;
            coordinatesAndDegrees(frameNumber, 3) = rotation;
            
            if params.reRef
                 coordinatesAndDegrees(frameNumber, 1:2) = ...
                     -coordinatesAndDegrees(frameNumber, 1:2);
                break
            end
            
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
            referenceFrame = imresize(referenceFrame, params.scalingFactor);
            break
        else
            continue
        end
        
    end
end


if ~isfield(params, 'reRef')
    coarseRefFrame = rotateCorrectedCoarse./counterArray;
    coarseRefFrame = Crop(coarseRefFrame);
    coarseRefFrame = double(coarseRefFrame)/255;
    save(outputFileName, 'coarseRefFrame');
else
    coarseRefFrame = [];
end

if params.enableVerbosity >= 1 && ~isfield(params, 'reRef')
    if isfield(params, 'axesHandles')
        axes(params.axesHandles(3));
        colormap(params.axesHandles(3), 'gray');
    else
        figure('Name', 'Coarse Reference Frame');
    end
    imshow(coarseRefFrame);
end

end