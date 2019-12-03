function [correlationMap, cache, xPeak, yPeak, peakValue] = ...
    FastStripCorrelation(strip, referenceFrame, cache, isGPU)
%FASTSTRIPCORRELATION Correlation computed using FFT.
%   strip, referenceFrame are required inputs.
%
%   cache is used to save calculations done before. The user of this
%   function is responsible for ensuring that an old cache is not provided
%   to a call with a different reference frame. Once finished, the cache is
%   returned for so the user can pass in back in next time. (default empty)
%
%   isGPU should be true if the GPU should be used. (default false)

%% Default args
if nargin < 3
   cache = struct; 
end

if nargin < 4
    isGPU = false;
end

gpuTask = getCurrentTask;
strip = WhereToCompute(strip, isGPU);
referenceFrame = WhereToCompute(referenceFrame, isGPU);

% precision of the computations
eps = 10^-6;

strip = single(strip);
referenceFrame = single(referenceFrame);

% get dimensions
[stripHeight, stripWidth] = size(strip);
[refHeight, refWidth] = size(referenceFrame);

if isempty(fieldnames(cache))
    % precomputed arrays
    mask = WhereToCompute(ones(stripHeight, stripWidth,'single'), isGPU);
    fuv = conv2(mask, referenceFrame);
    f2uv = conv2(mask, referenceFrame.^2);

    % energy of the reference
    euv = f2uv - (fuv.^2)/(stripHeight * stripWidth);
    euv(euv == 0) = eps;
    
    % shift, sqrt, take reciprocal
    cache.ieuv = 1./sqrt(circshift(euv,[0 0]));

    % fft of the reference
    cache.fr = fft2(padarray(referenceFrame,[stripHeight stripWidth]-1,0,'post'));
end

%% subtract the mean, compute energy, and fft
currentStripZero = strip - mean(strip(:));
currentStripEnergy = sqrt(sum(currentStripZero(:).^2));
ft = fft2(padarray(currentStripZero,[refHeight refWidth]-1,0,'pre'));

%% compute the normalized xcorr
correlationMap = ifft2(conj(ft).*(cache.fr)) .* cache.ieuv / currentStripEnergy;

%% find peak location and value
[xPeak, yPeak, peakValue] = FindPeak(correlationMap, isGPU);

