function success = Tester_FilterEyePosition

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample position file

    % the video resides under /demo folder.
    inputFile = FindFile('aoslo_demo_pos_960hz.mat');
    
    % load raw position and time
    load(inputFile,'timeSec','position');
    
    % since this example file contains uninverted position shifts, we need
    % to invert to get eye position. We also need to convert from pixel
    % units to visual degrees.
    pixelSizeDeg = 0.83 / 512;
    positionDeg = -position * pixelSizeDeg;
    
    %% First test
    
    % use default params and plot intermediate filtering stages
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 2;
    p.maxGapDurationMs = 50;
    [~, p] = FilterEyePosition([positionDeg timeSec], p);
    

    %% Second test
    
    % create a hidden file with desired data
    hiddenFile = '.demoPositionFile.mat';
    
    % use a filepath as input and only plot the final output
    save(hiddenFile,'positionDeg','timeSec');
    p.enableVerbosity = 1;
    [~, p] = FilterEyePosition(hiddenFile, p);
    delete(hiddenFile);
    delete(p.outputFilePath);
    
    
    %% third test
    
    % controlled failure: send in position in pixel units
    try 
        FilterEyePosition([position timeSec], p);
        success = false;
    catch
        success = true;
    end

catch 
    success = false;
end

warning(origState);