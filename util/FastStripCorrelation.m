function [correlationMap, cache] = FastStripCorrelation(strip, referenceFrame, cache, downSampleFactor, isGPU)
%FASTSTRIPCORRELATION Correlation computed using FFT.
%   strip, referenceFrame are required inputs.
%
%   cache is used to save calculations done before. The user of this
%   function is responsible for ensuring that an old cache is not provided
%   to a call with a different reference frame. Once finished, the cache is
%   returned for so the user can pass in back in next time. (default empty)
%
%   downSampleFactor is used to shrink everything during internal
%   computations. If downSampleFactor > 1, it is the factor to shrink by.
%   For example, downSampleFactor = 2 means to reduce everything to half
%   its size. If downSampleFactor < 1, each pixel is kept with that probability
%   (and otherwise it is zeroed out). (default 1)
%
%   isGPU should be true if the GPU should be used. (default false)

%% Default args
if nargin < 3
   cache = struct; 
end

if nargin < 4
    downSampleFactor = 1;
end

if nargin < 5
    isGPU = false;
end

% precision of the computations
eps = 10^-6;

if downSampleFactor > 1
    strip = imresize(single(strip) / 255 , 1 / downSampleFactor);
    referenceFrame = imresize(single(referenceFrame) / 255, 1 / downSampleFactor);
else
    strip = single(strip) / 255;
    strip = strip .* (rand(size(strip)) < downSampleFactor);
    referenceFrame = single(referenceFrame) / 255;
    referenceFrame = referenceFrame .* (rand(size(referenceFrame)) < downSampleFactor);
end

[stripHeight, stripWidth] = size(strip);
[refHeight, refWidth] = size(referenceFrame);

if isempty(fieldnames(cache))
    % precomputed arrays
    mask = WhereToCompute(ones(stripHeight, stripWidth,'single'), isGPU);
    fuv = conv2(referenceFrame, mask);
    f2uv = conv2(referenceFrame.^2, mask);

    % energy of the reference
    euv = (f2uv - (fuv.^2)/(stripHeight * stripWidth));
    euv(euv == 0) = eps;

    % shift, sqrt, and take the reciprocal of euv here for speed up. Note that
    % division is more expensive than multiplication.
    cache.ieuv = 1./sqrt(circshift(euv, -[stripHeight stripWidth]+1));

    % fft of the reference
    cache.cm = stripHeight + refHeight - 1;
    cache.cn = stripWidth  + refWidth  - 1;
    cache.fr = fft2(referenceFrame, cache.cm, cache.cn);
end

%% subtract the mean, compute energy, and fft
currentStripZero = strip - mean(strip(:));
currentStripEnergy = sqrt(sum(currentStripZero(:).^2));
ft = fft2(currentStripZero, cache.cm, cache.cn);

%% compute the normalized xcorr
correlationMap = ifft2(conj(ft).*(cache.fr)) .* cache.ieuv / currentStripEnergy;

end

