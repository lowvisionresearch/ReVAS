function [toneCurve, toneMapper] = GetToneCurve(grayLevels, gains, stopIndex)
% [toneCurve, toneMapper] = GetToneCurve(grayLevels, gains, stopIndex)
%
%
%

if nargin < 1 || isempty(grayLevels)
    grayLevels = [8 16 32 64];
end

if nargin < 2 || isempty(gains)
    gains = (grayLevels/255).^(-0.4);
end

if nargin < 3 || isempty(stopIndex)
    stopIndex = 200;
end

if any(grayLevels < 0 | grayLevels > 255)
    error('GetToneCurve: grayLevels must be between 1-255.');
end

if stopIndex > 255 || stopIndex < 1
    error('GetToneCurve: stopIndex must be between 1-255.');
end

if length(gains) ~= length(grayLevels)
    error('GetToneCurve: gains and grayLevels must have same lengths.');
end

zeroIx = find(grayLevels == 0);
if ~isempty(zeroIx)
    warning('GetToneCurve: gray level 0 will be removed.');
    gains(zeroIx) = [];
    grayLevels(zeroIx) = [];
end

[~,maxIx] = max(grayLevels);
if grayLevels(maxIx) == stopIndex
    gains(maxIx) = [];
    grayLevels(maxIx) = [];
end

% do a pieacewise fitting
x = linspace(0,1,256);
[grayLevels,ix] = sort(grayLevels);
gains = gains(ix);
toneFun = fit( [0 grayLevels stopIndex:255]'/255, ...
    [1 gains ones(1,256-stopIndex)]', 'pchip', 'Normalize', 'on' );
toneCurve = feval(toneFun,x)'.*x;

% check if it's nonmonotonic
if diff(toneCurve) < 0
    error('GetToneCurve: selected parameters resulted in a nonmonotonic tone curve.!');
end

% convert to index map
toneMapper = uint8(toneCurve*255);



