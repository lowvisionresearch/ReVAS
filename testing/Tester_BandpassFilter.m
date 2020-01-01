function success = Tester_BandpassFilter

% suppress warnings
origState = warning;
warning('off','all');


try
    %% read in sample video
    
    % the video resides under /testing folder.
    inputVideo = FindFile('tslo-dark.avi');    
    
    
    %% First test
    % use default params
    p = struct; 
    p.overwrite = true;

    % test with a video path with default settings 
    outputVideoPath = BandpassFilter(inputVideo, p); %#ok<*ASGLU>
    
    %% Second test
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    
    % assume some of the frames were bad
    clear p;
    p = struct;
    p.overwrite = true;
    p.smoothing = 0.5;
    p.lowSpatialFrequencyCutoff = 5;
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3]) = true;

    outputVideo = BandpassFilter(videoArray,p); %#ok<*NASGU>
    
    assert(size(videoArray,3) - size(outputVideo,3) == sum((p.badFrames)));
    
    %% Third test
    p.badFrames = false(3,1); % intentionally 3 frames
    p.badFrames([2 3]) = true;
    outputVideo2 = BandpassFilter(videoArray,p);
    
    assert(size(outputVideo2,3) == size(videoArray,3));
    
    success = true;
    
    %% cleanup
    delete(outputVideoPath);

catch err
    success = false;
end

warning(origState);

