function success = Tester_MakeReference

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample video
    
    % for tester, we're using an aoslo video, because tslo or larger fov
    % videos usually need to be preprocessed to remve within-frame
    % nonhomogeneities so that cross-correlation will be robust. 

    % the video resides under /testing folder.
    inputVideo = FindFile('aoslo.avi');
    
    % get the strip analysis results
    stripResults = load('aoslo_demo_pos.mat');
    
    
    %% First test
    
    % use default params
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 'frame';
    p.rowNumbers = stripResults.params.rowNumbers;
    p.oldStripHeight = stripResults.params.stripHeight;
    p.newStripWidth = 256;
    p.position = stripResults.position;
    p.timeSec = stripResults.timeSec;
    p.peakValueArray = stripResults.peakValueArray;
    [~,params, refFrame, outputFilePath] = MakeReference(inputVideo, p); %#ok<*ASGLU>
    delete(outputFilePath);
    
    %% Second test
    
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    p.newStripWidth = [];
    p.enhanceStrips = false;
    p.enableVerbosity = 'frame';
    [~, params, refFrame, outputFilePath] = MakeReference(videoArray, p);

    
    success = true;
catch err
    success = false;
end

warning(origState);


