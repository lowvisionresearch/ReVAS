function [coarseRefFrame, coordinatesAndDegrees] = RotateCorrect(shrunkFrames, bigFrames, ...
    referenceFrame, outputFileName, params)
%%ROTATE CORRECT      RotateCorrect takes in a frames and reference frame--
% both represented as 2D matrices--and returns a 4-column matrix. It
% rotates frames until they correlate with the reference frame with a 
% correlation value greater than or equal to the minimumPeakThreshold. If
% none of the rotations surpasses the threshold, then the function simply
% returns the optimal degree rotation for that frame.

%% Initialize variables
if ~isfield(params, 'degreeRange')
    params.degreeRange = 1;
end

if ~isfield(params, 'maximumPeakRatio')
    params.maximumPeakRatio = 0.7;
end

% usefulEyePositionTraces contains: Columns 1 & 2 (coordinates) and Column
% 3 (degree that returns the ideal rotation)
rotations = -params.degreeRange:0.1:params.degreeRange;
coordinatesAndDegrees = zeros(size(shrunkFrames, 3), 3);

% Pre-allocate the template frame
rotateCorrectedCoarse = zeros(size(bigFrames, 1)*2, size(bigFrames, 2)*2);
counterArray = rotateCorrectedCoarse;
a = [];
%% Examine each frame
%for frameNumber = 1:size(shrunkFrames, 3)
for frameNumber = 1:size(shrunkFrames, 3)
    
    frame = shrunkFrames(:, :, frameNumber);

    % First check if the frame has a correlation value greater than the
    % designated threshold. If so, then update the reference frame 
    % and move on to the next frame. Because StripAnalysis only accepts 3D
    % matricies, convert frame into 3D matrix first.
    threeDimensionalFrame = zeros(size(frame, 1), size(frame, 2), 2);
    threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + ...
        double(frame);

    % Templates cannot be the same, so add a random 1 to the second frame
    % in the third dimension
    threeDimensionalFrame(1, 1, 2) = 1;
    
    params.peakDropWindow = 25;
    [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
        threeDimensionalFrame, referenceFrame, params);
    
    peakRatio = statisticsStructure.peakRatios(1);
    usefulEyePositionTraces = usefulEyePositionTraces(1, :);

    if peakRatio <= params.maximumPeakRatio && ismember(1, ...
            ~isnan(usefulEyePositionTraces))
        coordinatesAndDegrees(frameNumber, 1:2) = usefulEyePositionTraces ...
            * 1/params.scalingFactor;
        coordinatesAndDegrees(frameNumber, 3) = 0;
        a = [a frameNumber];
        disp(peakRatio)
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
    else
        % If the correlation value does not pass the threshold, start 
        % rotating frames.

        for k = 1:max(size(rotations))
            rotation = rotations(k);
            
            % Already checked whether the frame without rotation exceeds
            % the threshold, so skip this iteration to reduce runtime
            if rotation == 0
                continue
            end
            
            frame = imrotate(frame, rotation);

            threeDimensionalFrame = zeros(size(frame, 1), size(frame, 2), 2);
            threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + ...
                double(frame);
            
            % Templates cannot be the same, so add a random 1 to the second frame
            % in the third dimension
            
            threeDimensionalFrame(1, 1, 2) = 1;
            params.peakDropWindow = 25;
            [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
                threeDimensionalFrame, referenceFrame, params);
            
            peakRatio = statisticsStructure.peakRatios(1);
            usefulEyePositionTraces = usefulEyePositionTraces(1, :);
            
            if isnan(usefulEyePositionTraces)
                continue
            end
            
            % If the peakValue for this rotation exceeds the threshold,
            % then move on to the next frame by breaking out of the
            % rotations for-loop. Otherwise, continue rotating
            if peakRatio <= params.maximumPeakRatio
                a = [a frameNumber];
                disp(peakRatio)
                coordinatesAndDegrees(frameNumber, 1:2) = ...
                    usefulEyePositionTraces * 1/params.scalingFactor;
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
                referenceFrame = imresize(referenceFrame, params.scalingFactor);
                break
            else
                continue
            end
        
        end
    end
end
disp(a)
coarseRefFrame = rotateCorrectedCoarse./counterArray;
coarseRefFrame = Crop(coarseRefFrame);
coarseRefFrame = double(coarseRefFrame)/255;

save(outputFileName, 'coarseRefFrame');

if params.enableVerbosity >= 1
    if isfield(params, 'axesHandles')
        axes(params.axesHandles(3));
        colormap(params.axesHandles(3), 'gray');
    else
        figure('Name', 'Coarse Reference Frame');
    end
    imshow(coarseRefFrame);
end

end