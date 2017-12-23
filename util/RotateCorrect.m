function [coarseRefFrame, coordinatesAndDegrees] = RotateCorrect(shrunkVideoPath, bigVideoPath, ...
    referenceFrame, outputFileName, parametersStructure)
%%ROTATE CORRECT     RotateCorrect checks each frame to see whether
%                    rotation is necessary (using peak ratios as the
%                    criterion). If the frame's peak ratio is above a
%                    specified threshold, then the function rotates the
%                    frame and cross-correlates until the ratio dips below 
%                    the threshold. It returns a coarseRefFrame using this
%                    correction method, as well as a matrix containing
%                    information regarding which degree rotation was used,
%                    and what each frame's peak ratio was.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |shrunkVideoPath| is the path to the scaled-down video created in 
%   CoarseRef (no default--must be specified)
%
%   |bigVideoPath| is the path to the original video (no default--must be
%   specified)
%
%   |referenceFrame| is the matrix representation of the temporary 
%   reference frame (no default--must be specified)
%
%   |outputFileName| is the filename of the output save file (no default--must 
%   be specified)
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   degreeRange         :   specifies the degree range to test for each 
%                          frame, passed in as a scalar (i.e., the function 
%                          will test rotations from -5 to +5 degrees if 
%                          user specifies 5). (default 5)
%   peakDropWindow      :   size of the correlation map to exclude when
%                          finding the second peak, for calculating peak 
%                          ratios. Units are in pixels. (default 25)
%   scalingFactor       :   the amount by which each frame is scaled down
%                          for the coarse cross-correlating
%                          0 < scalingFactor <= 1 (default 1)
%   rotateMaximumPeakRatio : specifies the peak ratio threshold below which 
%                           a rotated frame will be considered a "good" 
%                           frame (default 0.6)
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       videoPath = 'MyVid.avi';
%       load('MyVid_params.mat')
%
%       coarseParameters.degreeRange = 1;
%       coarseParameters.peakDropWindow = 25;
%       coarseParameters.rotateMaximumPeakRatio = 0.6;
%
%       frames = VideoPathToArray(videoPath);
%       refFrameNumber = round(size(frames, 3) / 2)
%       referenceFrame = frames(:, :, refFrameNumber)
%
%       coarseReferenceFrame = RotateCorrect(frames, frames,
%       referenceFrame, 'MyVid_coarseRef.mat', coarseParameters);

%% Initialize variables
if ~isfield(parametersStructure, 'degreeRange')
    parametersStructure.degreeRange = 5;
end

if ~isfield(parametersStructure, 'peakDropWindow')
    parametersStructure.peakDropWindow = 25;
end

bigVideoObject = VideoReader(bigVideoPath);
shrunkVideoObject = VideoReader(shrunkVideoPath);

% Preallocate the rotate corrected coarse reference frame and counter array
rotateCorrectedCoarse = zeros(bigVideoObject.Height*2.5, bigVideoObject.Height*2.5);
counterArray = rotateCorrectedCoarse;

% usefulEyePositionTraces contains: Columns 1 & 2 (coordinates) and Column
% 3 (degree that returns the ideal rotation)
rotations = -parametersStructure.degreeRange:0.1:parametersStructure.degreeRange;
totalFrames = shrunkVideoObject.Framerate * shrunkVideoObject.Duration;
coordinatesAndDegrees = zeros(totalFrames, 3);

%% Examine each frame
for frameNumber = 1:totalFrames
    
    smallFrame = double(readFrame(shrunkVideoObject))/255;
    if ndims(smallFrame) == 3
        smallFrame = rgb2gray(smallFrame);
    end
    bigFrame = double(readFrame(bigVideoObject))/255;
    if ndims(bigFrame) == 3
        bigFrame = rgb2gray(bigFrame);
    end
    % First check if the frame has a peak ratio lower than the
    % designated threshold. If so, then update the reference frame
    % and move on to the next frame. 
    [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
        smallFrame, referenceFrame, parametersStructure);
    
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
        maxRow = bigVideoObject.Height + minRow - 1;
        maxColumn = bigVideoObject.Width + minColumn - 1;
        
        selectRow = round(minRow):round(maxRow);
        selectColumn = round(minColumn):round(maxColumn);
        
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
        
        tempFrame = imrotate(smallFrame, rotation);
        
        [~, usefulEyePositionTraces, ~, statisticsStructure] = StripAnalysis(...
            tempFrame, referenceFrame, parametersStructure);
        
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
            maxRow = bigVideoObject.Height + minRow - 1;
            maxColumn = bigVideoObject.Width + minColumn - 1;
            
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
