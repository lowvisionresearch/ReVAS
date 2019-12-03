function success = Tester_Interpolation2D

%% load a sample data
onion   = rgb2gray(imread('onion.png'));
peppers = rgb2gray(imread('peppers.png'));
c = normxcorr2(onion,peppers);
c = circshift(c,310,1);

[xPeak, yPeak] = FindPeak(c, false);


try
    %% first test
    % basic operation with default params
    
    [xPeakNew, yPeakNew] = ...
        Interpolation2D(c, xPeak, yPeak, struct);

    
    
    success = true;
catch
    success = false;
end