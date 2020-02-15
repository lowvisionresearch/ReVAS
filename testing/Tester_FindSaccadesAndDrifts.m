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
    [~, p] = ...
        FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    
    %% Second test
    
    % use ivt method. it cannot find any saccades in this example!!!
    p.algorithm = 'ivt';
    [~,p2] = FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    assert(isempty(p2.saccades));

    %% third test
    
    % try EK algorithm. should result in similar outcome as hybrid.
    p.algorithm = 'ek';
    [~,p3] = FindSaccadesAndDrifts([filteredEyePositions timeSec],  p);
    
    %% fourth test
    
    % try with a file input
    [~,p4] = FindSaccadesAndDrifts(inputFile,  p);
    delete(p4.outputFilePath);
    
    assert(length(p3.saccades) == length(p4.saccades));
    
    success = true;
    
catch 
    success = false;
end

warning(origState);