function [xPeak, yPeak, peakValue] = FindPeak(correlationMap, isGPU)
%FIND PEAK Call this function after performing cross-correlation.
%   Call this function after performing cross-correlation in order to
%   identify the peak coordinates and value.

if nargin < 2 || isempty(isGPU) || (gpuDeviceCount==0)
    isGPU = false;
end


%% Find peak

% Find peak of correlation map
[peakValue, argmax] = max(correlationMap(:));
xPeak = ceil(argmax/size(correlationMap, 1));
yPeak = mod(argmax - 1, size(correlationMap, 1)) + 1;

% If there is a tie for max peak, get the one which is closest to the left
% boundary of the reference frame.
if size(xPeak,1) > 1
    [xPeak,ix] = min(xPeak); 
    yPeak = yPeak(ix);
end

% if GPU is enabled, only return the peak location and value not the map.
if isGPU
    peakValue = gather(peakValue);
    xPeak = gather(xPeak);
    yPeak = gather(yPeak);
end

 