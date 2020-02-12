function [xPeakNew, yPeakNew, peakValueNew] = ...
    Interpolation2D(correlationMap2D, xPeak, yPeak, neighborhoodSize, ...
        subpixelDepth, interpMethod, enableGPU)
%2D INTERPOLATION Completes 2D Interpolation on a correlation map.
%   Completes 2D Interpolation on a correlation map and returns the new
%   interpolated peak coordinates. Uses the |spline| option in |interp2|.
%   -----------------------------------
%   Input
%   -----------------------------------
%   |correlationMap2D| is a 2D array containing the output of normalized
%   cross-correlation operation.
%
%   |xPeak| and |yPeak| are peak coordinates.
%
%   'neighborhoodSize' in pixels.
%
%   'subpixelDepth' is an integer, intigating how many octave levels deep
%   we want to interpolate. (default 3, i.e., 2^-3 = 0.125px)
%
%   'interpFunc' is a function pointer. e.g., @interp2, OR  @splinterp2
%
%   'enableGPU' is a flag used to determine whether the correlation map is 
%   in GPU memory.
%


if nargin < 3
    error('Interpolation2D: insufficient number of input arguments.');
end

if isempty(correlationMap2D)
    error('Interpolation2D: correlation map is empty');
end

if ~all(IsPositiveInteger([xPeak yPeak]))
    error('Interpolation2D: xPeak and yPeak must be integers.');
end

%% Set parameters to defaults if not specified.

if nargin < 4 || isempty(neighborhoodSize)
    neighborhoodSize = 5; 
end

if nargin < 5 || isempty(subpixelDepth)
    subpixelDepth = 2; 
end

if nargin < 5 || isempty(interpMethod)
    interpMethod = 'spline'; 
end

if nargin < 5 || isempty(enableGPU)
    enableGPU = false; 
end


%%
% get sizes
[refHeight, refWidth] = size(correlationMap2D);

% check if we hit the boundaries of the correlation map
left = max(1, xPeak - neighborhoodSize);
right = min(refWidth, xPeak + neighborhoodSize);
top = max(1, yPeak - neighborhoodSize);
bottom = min(refHeight, yPeak + neighborhoodSize);

% new resolution
dpx = (2^(-subpixelDepth));

% crop out subsection of the correlation map
subCorrMap = correlationMap2D(top:bottom,left:right);
[subHeight, subWidth] = size(subCorrMap);

% if it's in GPU, bring it 
if enableGPU
    subCorrMap = gather(subCorrMap);
end

% create fine meshgrid
[xq, yq] = meshgrid(1:dpx:subWidth, 1:dpx:subHeight);

% now interpolate
fineSubCorrMap = interp2(subCorrMap,xq,yq,interpMethod);

% find peak
[xPeakNew, yPeakNew, peakValueNew] = FindPeak(fineSubCorrMap, false);

% get half size of the finer map
[fineSubHeight, fineSubWidth] = size(fineSubCorrMap);

% add offset to make coordinates with respect to the full correlation map
xPeakNew = xPeakNew * (subWidth+1) / (fineSubWidth+1) + left - 1;
yPeakNew = yPeakNew * (subHeight+1) / (fineSubHeight+1) + top - 1;





