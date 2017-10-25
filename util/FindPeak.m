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
offset = 0;
originalCorrelationMap = correlationMap;
if isfield(parametersStructure, 'searchWindowPercentage')
    offset = floor(size(correlationMap, 2) * (1 - parametersStructure.searchWindowPercentage)/2);
    correlationMap = ...
        correlationMap(:,offset+1 : offset+ceil(size(correlationMap, 2) * parametersStructure.searchWindowPercentage));
end

% Relevant only if gaussian filtering not enabled.
% This determines the window around the first peak which is to be
% disregarded before searching for the second peak.
if isfield(parametersStructure, 'peakDropWindow')
   peakDropWindow = parametersStructure.peakDropWindow;
else
   peakDropWindow = parametersStructure.stripHeight;
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
        
        % If there is a tie for max peak, choose the one closest to the
        % center of the correlation map.
        if size(xPeak,1) > 1 && size(yPeak,1) > 1
            indexOfClosest = 1;
            closestDistance = Inf;
            for i = 1:size(xPeak)
                dist = (yPeak-size(correlationMap,1))^2 + (xPeak-size(correlationMap,2))^2;
                if dist < closestDistance
                    closestDistance = dist;
                    indexOfClosest = i;
                end
            end

            xPeak = xPeak(indexOfClosest);
            yPeak = yPeak(indexOfClosest);
        end
            
        peakValue = correlationMap(yPeak, xPeak);
        
        % We do not use the second peak when using Gaussian approach
        % because we do not care about peak ratio in this case.
        secondPeakValue = NaN;
        
    else
        % Find peak of correlation map
        [yPeak, xPeak] = find(correlationMap==max(correlationMap(:)));
        
        % If there is a tie for max peak, arbitrarily break it since the
        % second highest peak will be the same and this frame will be
        % thrown out anyways.
        if size(xPeak,1) > 1
           xPeak = xPeak(1);
        end
        if size(yPeak,1) > 1
            yPeak = yPeak(1);
        end
            
        peakValue = correlationMap(yPeak, xPeak);

        % Find second highest point of correlation map
        peakWindowMinX = max(1, xPeak - peakDropWindow);
        peakWindowMaxX = min(size(correlationMap, 2), xPeak + peakDropWindow);
        peakWindowMinY = max(1, yPeak - peakDropWindow);
        peakWindowMaxY = min(size(correlationMap, 1), yPeak + peakDropWindow);
        for y = (peakWindowMinY:peakWindowMaxY)
            for x = (peakWindowMinX:peakWindowMaxX)
                correlationMap(y, x) = -inf; % remove highest peak
            end
        end
        secondPeakValue = max(correlationMap(:));
    end
    
    if isfield(parametersStructure, 'searchWindowPercentage')
       % Add back in the offset removed in the beginning if applicable.
       xPeak = xPeak + offset;
       
       % If results using this searchWindowPercentage are not acceptable,
       % try again with a larger searchWindow.
       isAcceptable = true;
       if peakValue < parametersStructure.minimumPeakThreshold
           isAcceptable = false;
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
            if fitOutput.c1 > parametersStructure.maximumSD
                isAcceptable = false;
            end
            fitOutput = fit(((1:size(middleCol,1))-ceil(size(middleCol,1)/2))', middleCol, 'gauss1');
            if fitOutput.c1 > parametersStructure.maximumSD
                isAcceptable = false;
            end
       else
           % Check peak ratio if not using gaussian filtering.
           if secondPeakValue/peakValue > parametersStructure.maximumPeakRatio
               isAcceptable = false;
           end
       end
       
       % If the above checks found that this result was not acceptable and
       % we can expand the searchWindowPercentage, run FindPeak again with
       % a larger searchWindowPercentage via recursion.
       if parametersStructure.searchWindowPercentage < 1 && ~isAcceptable
           largerSearchWindowParams = parametersStructure;
           largerSearchWindowParams.searchWindowPercentage = ...
               min(parametersStructure.searchWindowPercentage + 0.1, 1);
           [xPeak, yPeak, peakValue, secondPeakValue] = ...
               FindPeak(originalCorrelationMap, largerSearchWindowParams);
       end
    end
end

