function [xPeak, yPeak, peakValue, secondPeakValue] = ...
    FindPeak(correlationMap, parametersStructure)
%FIND PEAK Call this function after performing |normxcorr2|.
%   Call this function after performing |normxcorr2| in order to
%   identify the peak coordinates and value (and second peak value, if
%   appropriate) using the method specific in the parameters structure (either
%   through Gaussian filtering or by searching after temporarily removing the
%   region around the first peak). Another option is to pass in the
%   estimated frame shifts from CoarseRef. Doing so will allow the function
%   to find search windows around each strip according to the approximate
%   shifts. It will then search for a peak in those windows.

% Use the appropriate method to identify the peak

%     if isfield(parametersStructure, 'roughEyePositionTraces')
%         % First just grab all variables from parametersStructure and rename
%         % them with shorter names for readaibility
%         yWindow = round(parametersStructure.windowSize(1)/2);
%         xWindow = round(parametersStructure.windowSize(2)/2);
%         stripHeight = parametersStructure.stripHeight;
%         stripsPerFrame = parametersStructure.stripsPerFrame;
%         stripNumber = parametersStructure.stripNumber;
%         positions = parametersStructure.roughEyePositionTraces;
%         stripWidth = parametersStructure.stripWidth;
%         
%         currStrip = mod(stripNumber, stripsPerFrame);
%         currFrame = ceil(stripNumber/stripsPerFrame);
        
        % roughEyePositionTraces is from CoarseRef which gives the
        % approximate shift of the entire frame. This next few lines
        % assume that the frame shift applies to all strips in that frame
        % (for example, if the frame shift was [-10, -10], we assume that
        % all strips for that frame were shifted [-10, -10]).
%         rowShift = positions(currFrame, 1);
%         columnShift = positions(currFrame, 2);
%         rowCoordinate = ((currStrip-1)*stripHeight) + rowShift + stripHeight;
%         columnCoordinate = stripWidth + columnShift;
%         disp(rowCoordinate)
%         disp(columnCoordinate)
%         window = [rowCoordinate-yWindow:rowCoordinate+yWindow, columnCoordinate-xWindow:...
%             columnCoordinate+xWindow];
%         [yPeak, xPeak] = find(correlationMap==max(correlationMap(window)));
%         peakValue = correlationMap(yPeak, xPeak);
%         secondPeakValue = -inf;
%         
    if parametersStructure.enableGaussianFiltering
        % Apply Gaussian Filter and get difference between correlations
        gaussianFilteredCorrelation = ...
            imgaussfilt(correlationMap, ...
            parametersStructure.gaussianStandardDeviation);
        gaussianDifferenceCorrelation = ...
            correlationMap - gaussianFilteredCorrelation;
        
        % Find peak of correlation map
        [yPeak, xPeak] = find(gaussianDifferenceCorrelation==max(gaussianDifferenceCorrelation(:)));
        peakValue = correlationMap(yPeak, xPeak);
        
        % We do not use the second peak when using Gaussian approach
        % because we do not care about peak ratio in this case.
        secondPeakValue = -inf;
        
    else
        % Find peak of correlation map
        [yPeak, xPeak] = find(correlationMap==max(correlationMap(:)));
        peakValue = max(correlationMap(:));

        % Find second highest point of correlation map
        peakWindowMinX = max(1, xPeak - parametersStructure.stripHeight);
        peakWindowMaxX = min(size(correlationMap, 2), xPeak + parametersStructure.stripHeight);
        peakWindowMinY = max(1, yPeak - parametersStructure.stripHeight);
        peakWindowMaxY = min(size(correlationMap, 1), yPeak + parametersStructure.stripHeight);
        for y = (peakWindowMinY:peakWindowMaxY)
            for x = (peakWindowMinX:peakWindowMaxX)
                correlationMap(y, x) = -inf; % remove highest peak
            end
        end
        secondPeakValue = max(correlationMap(:));
    end

end

