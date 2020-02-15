function [correlationMap, xPeak, yPeak, peakValue, cache] = ...
    LocateStrip(thisStrip,params,cache)
%LOCATE STRIP 
%[correlationMap, xPeak, yPeak, peakValue, varargout] = ...
%     LocateStrip(thisStrip,params,cache)
%
%   Locate a strip on a reference frame using a selected cross-correlation
%   method.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |thisStrip| is a horizontal strip, i.e., a 2D array.
%
%   |params| is a struct as specified below.
% 
%   |cache| is a struct needed in 'fft' method. It allows for re-using
%   precomputed matrices within that method.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------

%   referenceFrame         : a 2D array, representing a reference frame.
%   corrMethod             : method to use for cross-correlation. you can
%                            choose from 'normxcorr' for matlab's built-in
%                            normxcorr2, 'mex' for opencv's correlation, or
%                            'fft' for our custom-implemented fast
%                            correlation method. 'cuda' is placed but not
%                            implemented yet (default 'mex')
%   enableGPU              : a logical. if set to true, use GPU. (works for
%                            'mex' method only.
%   adaptiveSearch         : a logical. if set to true, fft method does not
%                            use cache.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |correlationMap|       : cross-correlation matrix. It could be a
%                            gpuArray in 'enableGPU' is set to true.
%   |xPeak|                : horizontal location of the peak. i.e.,
%                            x coordinate in the reference frame.
%   |yPeak|                : vertical location of the peak.
%   |peakValue|            : peak cross-correlation value. (must be 
%                            between -1 and 1).
%   |cache|                : meant to pass 'cache' struct to output for
%                            later re-use within the 'fft' method.
%
%
%


% check for stripWidth. If it's larger than refFrame width, crop.
w = size(params.referenceFrame,2);
if size(thisStrip,2) > w
    thisStrip(:,(w+1):end) = [];
end

switch params.corrMethod
    case 'mex'
        if params.enableGPU % TO-DO: revise matchTemplateOCV_GPU to return peak location and value 
            correlationMap = matchTemplateOCV_GPU(thisStrip, params.referenceFrame(params.rowStart:params.rowEnd,:)); 
            [xPeak, yPeak, peakValue] = FindPeak(correlationMap, params.enableGPU);
        else
            [correlationMap,xPeak,yPeak,peakValue] = ...
                matchTemplateOCV(thisStrip, params.referenceFrame(params.rowStart:params.rowEnd,:)); 
        end

    case 'normxcorr'
        correlationMap = normxcorr2(thisStrip, params.referenceFrame(params.rowStart:params.rowEnd,:)); 
        [xPeak, yPeak, peakValue] = FindPeak(correlationMap, params.enableGPU);

    case 'fft'
        if params.adaptiveSearch
            cache = struct;
        end
        [correlationMap, cache,xPeak,yPeak,peakValue] = ...
            FastStripCorrelation(thisStrip, params.referenceFrame(params.rowStart:params.rowEnd,:), cache, params.enableGPU);
        
    case 'cuda'
        [corrmap,xPeak,yPeak,peakValue,~] = cuda_match(single(thisStrip),params.copyMap);
        if params.copyMap
            correlationMap = fftshift(corrmap);
            correlationMap = correlationMap(1:params.outsize(1),1:params.outsize(2));
        else
            correlationMap = corrmap;
        end
        
        % casting needed
        xPeak = double(xPeak);
        yPeak = double(yPeak);

    otherwise
        error('LocateStrip: unknown cross-correlation method.');
end




