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

% Cut out smaller correlation map to search in if applicable. Doing this
% will essentially restrict searching to the center area and any false
% peaks near the edges will be ignored.
yOffset = 0;
xOffset = 0;
if isfield(parametersStructure, 'searchWindowPercentage')
    yOffset = floor(size(correlationMap, 1) * (1 - parametersStructure.searchWindowPercentage)/2);
    xOffset = floor(size(correlationMap, 2) * (1 - parametersStructure.searchWindowPercentage)/2);
    correlationMap = correlationMap( ...
        yOffset+1 : yOffset+ceil(size(correlationMap, 1) * parametersStructure.searchWindowPercentage), ...
        xOffset+1 : xOffset+ceil(size(correlationMap, 2) * parametersStructure.searchWindowPercentage));
end

% Use the appropriate method to identify the peak
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
    
    % Add back in the offset removed in the beginning if applicable.
    if isfield(parametersStructure, 'searchWindowPercentage')
       yPeak = yPeak + yOffset;
       xPeak = xPeak + xOffset;
    end


end

