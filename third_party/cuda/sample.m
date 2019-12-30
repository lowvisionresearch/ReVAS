%% Compiling
mexcuda -lcufft -lcublas cuda_match.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu

%% Code to be included in StripAnalysis.m
%% Setup: call once before big loop

referenceFrame = referenceFrame-mean(mean(referenceFrame));
paddedReference = mean(mean(referenceFrame))*single(ones(size(referenceFrame)+[stripHeight,stripWidth]));
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

% initialize matching:
cuda_match(paddedReference,localVars,stripHeight,stripWidth)
copyMap = true;

%% Within loop:
case 'cuda'
    [corrmap,xloc,yloc,peak,secondPeak] = cuda_match(strip,copyMap);
    outsize = size(paddedReference);
    if copyMap
        correlationMap = fftshift(corrmap);
        correlationMap = correlationMap(1:outsize(1)-1,1:outsize(2)-1);
        correlationMap(xloc,yloc) = peak;
        correlationMap = correlationMap/max(correlationMap(:)); %TODO: fix normalization
    end