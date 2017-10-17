function [correlationValues] = RotateCorrect(frames, referenceFrame, params)
%ROTATE CORRECT      RotateCorrect takes in a frames and reference frame--
% both represented as 2D matrices--and returns a 4-column matrix. The first
% column is the ideal degree rotation, the second column is the correlation
% value associated with that column, the third column is the x position
% (column number) and the fourth column is the y position (row number)

if ~isfield(params, 'degreeRange')
    params.degreeRange = 1;
end

% First column will store the optimal degree correction, second column will 
% store the associated correlation value, third column and fourth columns 
% will store the x and y positions (column and row positions), respectively
rotations = -degreeRange:0.1:degreeRange;
correlationValues = zeros(size(frames, 3), 4);

rotatedFrames = zeros(size(frames));

for k = 1:size(rotations, 2)
    degree = rotations(1, k);
    for frameNumber = 1:size(frames, 3)
        frame = frames(:, :, frameNumber);
        tempFrame = imrotate(frame, degree);
        rotatedFrames(:, :, tempFrame) = tempFrame;
    end
    
    [~, usefulEyePositionTraces, ~, statisticsStructure] = ...
        StripAnalysis(rotatedFrames, referenceFrame, params);
    
    tempCorrelationValues = statisticsStructure.peakValues;
    
    % Set all NaNs to 0, in order to prevent error'ing when comparing
    % new correlation values to previous NaNs
    NaNindices = isnan(tempCorrelationValues);
    tempCorrelationValues(NaNindices, 1) = 0;
    
    if k == 1
        correlationValues(:, 1) = degree;
        correlationValues(:, 2) = tempCorrelationValues;
        correlationValues(:, 3:4) = usefulEyePositionTraces;
       
    else
        % Find the frames in which the correlation value is higher with
        % this degree correction. Replace the information in all 4 columns
        % with the information on this iteration, since this degree
        % correction is more accurate.
        indices = find(correlationValues(:, 2) < tempCorrelationValues);
        correlationValues(indices, 2) = tempCorrelationValues(indices, 1);
        correlationValues(indices, 1) = degree;
        correlationValues(indices, 3:4) = usefulEyePositionTraces(indices, :);
    end
end


end