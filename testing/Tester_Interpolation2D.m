function success = Tester_Interpolation2D

try
    %% first test
    % basic operation with default params
    
    c = [0 .1 .1 .1 0;
        .1 .2 .4 .3 .1;
        .3 .3 .5 .4 .3;
        .2 .2 .1 .2 .1'
        0  .1 .2 .1 0];
    xPeak = 3; yPeak = 3;
    
    [xPeakNew, yPeakNew] = ...
        Interpolation2D(c, xPeak, yPeak);

    assert(round(xPeakNew * 3,3) == 10 & round(yPeakNew * 3,3) == 8 );
    
    
    %% second test
    % peak replicas
    c = [0 .1 .1 .1 0;
        .1 .2 .5 .5 .1;
        .3 .3 .5 .5 .3;
        .2 .2 .1 .2 .1'
        0  .1 .2 .1 0];
    xPeak = 3; yPeak = 3;
    
    [xPeakNew, yPeakNew] = ...
        Interpolation2D(c, xPeak, yPeak, [],[],'makima',[]);

    assert(round(xPeakNew * 3,3) == 11 & round(yPeakNew * 3,3) == 7 );
    
    
    %% third test
    % no peak
    c = zeros(5,5);
    xPeak = 5; yPeak = 5;
    
    [xPeakNew, yPeakNew] = ...
        Interpolation2D(c, xPeak, yPeak);
    
    assert(round(xPeakNew * 3,3) == 1 & round(yPeakNew * 3,3) == 1 );
    
    success = true;
catch
    success = false;
end