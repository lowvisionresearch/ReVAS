function cuda_prep(referenceFrame,stripHeight,stripWidth,firstTime)

% pad the reference frame
referenceFrame = single(referenceFrame);
[refHeight,refWidth] = size(referenceFrame);
padsize = size(referenceFrame)+[stripHeight,stripWidth];
paddedReference = single(zeros(padsize));
paddedReference(1:refHeight,1:refWidth) = referenceFrame;

% calculate local sums of reference
scratch = padarray(referenceFrame,[stripHeight,stripWidth]);
scratch = cumsum(scratch,1);
scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
scratch = cumsum(scratch,2);
localSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);

% calculate local variances of reference
scratch = padarray(referenceFrame.^2,[stripHeight,stripWidth]);
scratch = cumsum(scratch,1);
scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
scratch = cumsum(scratch,2);
localSquaredSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);
localVars = sqrt(localSquaredSums-localSums.^2/(stripHeight*stripWidth));
localVars(localVars==0) = max(max(localVars));

% invert, pad, and shift local variances
invLocalVars = 1./localVars;
paddedInvVars = mean(invLocalVars(:))*single(ones(cufftsize(padsize)));
paddedInvVars(1:padsize(1)-1,1:padsize(2)-1) = invLocalVars;
paddedInvVars = fftshift(paddedInvVars);
 
% initialize match
cuda_match(paddedReference,paddedInvVars,stripHeight,stripWidth,firstTime)

function [x, y] = cufftsize(size)
    if 2^ceil(log2(size(1))) <= 1024
        x = 2^ceil(log2(size(1)));
    else
        x = 512*ceil(size(1)/512);
    end
    if 2^ceil(log2(size(2))) <= 1024
        y = 2^ceil(log2(size(2)));
    else
        y = 512*ceil(size(2)/512);
    end 