function success = Tester_Pixel2Degree

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample position file

    % the video resides under /demo folder.
    inputFile = FindFile('aoslo_demo_pos_960hz.mat');
    
    % load raw position and time
    load(inputFile,'timeSec','position');
    
    %% First test
    
    % use default params and plot intermediate filtering stages
    p = struct; 
    p.fov = 0.83;
    p.frameWidth = 512;
    [outArg, p] = Pixel2Degree([position timeSec], p);
    

    %% Second test
    
    % create a hidden file with desired data
    hiddenFile = '.demoPositionFile.mat';
    
    % use a filepath as input and only plot the final output
    save(hiddenFile,'position','timeSec');
    [~, p] = Pixel2Degree(hiddenFile, p);
    
    % load positionDeg
    load(hiddenFile,'positionDeg');
    
    % compare with outArg
    nonnan = ~isnan(positionDeg(:,1));
    assert(all(all(outArg(nonnan,1:end-1) == positionDeg(nonnan,:))))
    
    % clean up
    delete(hiddenFile);
    delete(p.outputFilePath);
    
    
    success = true;
catch 
    success = false;
end

warning(origState);