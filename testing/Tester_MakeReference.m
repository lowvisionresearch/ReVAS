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
    inputVideo = 'aoslo.avi';

    str = which(inputVideo);
    if isempty(str)
        success = false;
        return;
    else
        [filepath,name,ext] = fileparts(str);
        inputVideo = [filepath filesep inputVideo];
    end
    
    % get the strip analysis results
    stripResults = load('aoslo_demo_pos.mat');
    
    
    %% First test
    
    % use default params
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 1;
    p.rowNumbers = stripResults.params.rowNumbers;
    p.oldStripHeight = stripResults.params.stripHeight;
    p.positions = stripResults.position;
    p.timeSec = stripResults.timeSec;
    p.peakValues = stripResults.peakValueArray;
    [refFrame, outputFilePath, params] = MakeReference(inputVideo, p);
    delete(outputFilePath);
    
    %% Second test
    
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    
    p.enhanceStrips = false;
    [refFrame, outputFilePath, params] = MakeReference(videoArray, p);

    
    success = true;
catch 
    success = false;
end

warning(origState);


