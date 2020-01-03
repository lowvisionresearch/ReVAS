function success = Tester_FindSaccadesAndDrifts

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample position file

    % the video resides under /demo folder.
    inputFile = FindFile('aoslo_demo_pos_540hz.mat');
    
    % load raw position and time
    load(inputFile,'timeSec','filteredEyePositions');
    
    %% First test
    
    % use default params (uses hybrid method)
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 1;
    [sHybrid, ~, ~, p] = ...
        FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    
    %% Second test
    
    % use ivt method. it cannot find any saccades in this example!!!
    p.algorithm = 'ivt';
    sIvt = FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    assert(isempty(sIvt));

    %% third test
    
    % try EK algorithm. should result in similar outcome as hybrid.
    p.algorithm = 'ek';
    sEk = FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    %% fourth test
    
    % try with a file input
    [sEk2,~,~,p] = FindSaccadesAndDrifts(inputFile,  p);
    delete(p.outputFilePath);
    
    assert(length(sEk) == length(sEk2));
    
    success = true;
    
catch 
    success = false;
end

warning(origState);