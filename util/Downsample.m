function img = Downsample(img, downSampleFactor)
%DOWNSAMPLE Downsamples the provided strip and referenceFrame according to
%the indicated downsample factor (see downSampleFactor description below).
%   img is either a strip or reference frame to be downsampled.
%
%   downSampleFactor is used to shrink everything during internal
%   computations. If downSampleFactor > 1, it is the factor to shrink by.
%   For example, downSampleFactor = 2 means to reduce everything to half
%   its size. If downSampleFactor < 1, every other pixel of the reference
%   frame is kept (in a checkerboard-like pattern). (default 1)

if nargin < 2
   return; 
end

if downSampleFactor == 1
    return;
elseif downSampleFactor > 1
    img = imresize(img , 1 / downSampleFactor);
else
    % source: http://matlabtricks.com/post-31/three-ways-to-generate-a-checkerboard-matrix).
    checkerboard = bsxfun(@xor, mod(1 : size(img, 1), 2)', mod(1 : size(img, 2), 2));
    img = img .* uint8(checkerboard);
end
end
