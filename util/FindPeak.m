function [xPeak, yPeak, peakValue] = FindPeak(correlationMap, isGPU)
%FIND PEAK Call this function after performing cross-correlation.
%   Call this function after performing cross-correlation in order to
%   identify the peak coordinates and value.

if nargin < 2 || isempty(isGPU) 
    isGPU = false;
end


%% Find peak

% Find peak of correlation map
peakValue = max(correlationMap(:));
[yPeak, xPeak] = find(correlationMap == peakValue,1);

% if GPU is enabled, only return the peak location and value not the map.
if isGPU
    peakValue = gather(peakValue);
    xPeak = gather(xPeak);
    yPeak = gather(yPeak);
end

 