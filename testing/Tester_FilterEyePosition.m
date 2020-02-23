function success = Tester_FilterEyePosition

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample position file

    % the video resides under /demo folder.
    inputFile = FindFile('aoslo_demo_pos.mat');
    
    % load raw position and time
    load(inputFile,'timeSec','positionDeg');
    
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
    p.axesHandles = [];
    [~, p] = FilterEyePosition(hiddenFile, p);
    delete(hiddenFile);
    delete(p.outputFilePath);
    
    
    success = true;
catch 
    success = false;
end

warning(origState);