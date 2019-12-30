function [correlationMap, xPeak, yPeak, peakValue] = CUDACorrelation(thisStrip, referenceFrame, firstTime, copyMap, gpuDeviceHandle)
%[correlationMap, xPeak, yPeak, peakValue] = CUDACorrelation(thisStrip, referenceFrame, firstTime, copyMap, gpuDeviceHandle)
%
%CUDACorrelation normalized cross-correlation computed using CUDA enabled GPU.
%
%   cache is used to save calculations done before. The user of this
%   function is responsible for ensuring that an old cache is not provided
%   to a call with a different reference frame. Once finished, the cache is
%   returned for so the user can pass in back in next time. (default empty)
%
%   isGPU should be true if the GPU should be used. (default false)

%% Default args
if nargin < 3 || isempty(firstTime)
   firstTime = true; 
end

if nargin < 4 || isempty(copyMap)
    copyMap = false;
end

if nargin < 5 || isempty(gpuDeviceHandle)
    gpuDeviceHandle = gpuDevice(1);
end

% if first time here, prepare padded ref
if firstTime
    reset(gpuDeviceHandle);
    
    [refHeight, refWidth] = size(referenceFrame);
    [stripHeight, stripWidth] = size(thisStrip);
    mu = mean(mean(referenceFrame));
    referenceFrame = referenceFrame - mu;
    paddedReference = mu * single(ones(size(referenceFrame)+[stripHeight,stripWidth]));
    paddedReference(1:refHeight,1:refWidth) = referenceFrame;

    scratch = cumsum(paddedReference,1);
    scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
    scratch = cumsum(scratch,2);
    localSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);

    scratch = cumsum(paddedReference.^2,1);
    scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
    scratch = cumsum(scratch,2);
    localSquaredSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);
    localVars = sqrt(localSquaredSums-localSums.^2/(stripHeight*stripWidth));
    localVars(localVars==0) = max(max(localVars));
    
    % first call to initialize
    cuda_match(paddedReference,localVars,stripHeight,stripWidth)   
end

[correlationMap,xPeak,yPeak,peakValue] = cuda_match(thisStrip,copyMap);
% offset = round(size(correlationMap)/2);
% cHeight = stripHeight + refHeight - 1;
% cWidth = stripWidth + refWidth - 1;
% xPeak = xPeak - offset(2);
% yPeak = yPeak - offset(1);




