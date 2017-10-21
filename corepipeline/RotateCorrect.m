function [correlationValues] = RotateCorrect(shrunkFrames, bigFrames, ...
    referenceFrame, params)
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

if ~isfield(params, 'minimumPeakThreshold')
    params.minimumPeakThreshold = 0.25;
end

% usefulEyePositionTraces contains: Columns 1 & 2 (coordinates) and Column
% 3 (degree that returns the ideal rotation)
rotations = -params.degreeRange:0.1:params.degreeRange;
coordinatesAndDegrees = zeros(size(shrunkFrames, 3), 3);

% Pre-allocate the template frame
rotateCorrectedCoarse = zeros(size(bigFrames, 1)*2, size(bigFrames, 2)*2);
counterArray = rotateCorrectedCoarse;

%% Examine each frame
for frameNumber = 1:size(shrunkFrames, 3)
    
    frame = shrunkFrames(:, :, frameNumber);
    
    % First check if the frame has a correlation value greater than the
    % designated threshold. If so, then update the reference frame 
    % and move on to the next frame.
    [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
        frame, referenceFrame, params);
    peakValue = statisticsStructure.peakValues(1);
    
    if peakValue >= params.minimumPeakThreshold
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
        
        bigFrame = bigFrames(:, :, frameNumber);
        
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
            
            [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
                frame, referenceFrame, params);
            peakValue = statisticsStructure.peakValues(1);
            
            % If the peakValue for this rotation exceeds the threshold,
            % then move on to the next frame by breaking out of the
            % rotations for-loop. Otherwise, continue rotating
            if peakValue >= params.minimumPeakThreshold
                coordinatesAndDegrees(frameNumber, 1:2) = ...
                    usefulEyePositionTraces * 1/params.scalingFactor;
                coordinatesAndDegrees(frameNumber, 3) = rotation;
                
                coordinates = coordinatesAndDegrees(frameNumber, 1:2);
                coordinates = round(ScaleCoordinates(coordinates));
                
                minRow = coordinates(1, 2);
                minColumn = coordinates(1, 1);
                maxRow = size(bigFrames, 1) + minRow - 1;
                maxColumn = size(bigFrames, 2) + minColumn - 1;
                
                selectRow = round(minRow):round(maxRow);
                selectColumn = round(minColumn):round(maxColumn);
                
                bigFrame = bigFrames(:, :, frameNumber);
                bigFrame = imrotate(bigFrame, rotation);
                
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

end