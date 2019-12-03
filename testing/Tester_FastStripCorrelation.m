function success = Tester_FastStripCorrelation


try 
    %% read test images
    onion   = rgb2gray(imread('onion.png'));
    peppers = rgb2gray(imread('peppers.png'));

    cRef = normxcorr2(onion,peppers);
    [xPeakRef, yPeakRef, peakValueRef] = FindPeak(cRef, false);

    %% first test
    % check for accuracy
    [correlationMap, cache, xPeak, yPeak, peakValue] = ...
        FastStripCorrelation(onion, peppers, struct, false);
    
    assert(abs(xPeak - xPeakRef)<=1);
    assert(abs(yPeak - yPeakRef)<=1); 
    assert(abs(peakValueRef-peakValue)<0.01);
    assert(rms(correlationMap(:) - cRef(:)) <= 0.0133);
    
    %% second test
    % check for cache usage
    correlationMap2 = FastStripCorrelation(onion, peppers, cache, false);
    
    assert(sum(correlationMap(:) - correlationMap2(:))==0);
    
    success = true;
    
catch
    success = false;
end