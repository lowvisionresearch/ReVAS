function success = Tester_StripAnalysis

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
    
    
    %% First test
   
    % use fullpath to a video
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 'frame';
    p.minPeakThreshold = 0.4;
    p.adaptiveSearch = true;
    [~, p, position1, timeSec, rawPosition, peakValueArray] = ...
        StripAnalysis(inputVideo, p); %#ok<*ASGLU>
    delete(p.outputFilePath);
  
    
    %% Second test
    
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    
    % use fft method
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = 'video';
    p.minPeakThreshold = 0.4;
    p.stripWidth = 128;
    p.adaptiveSearch = false;
    p.badFrames = false(32,1);
    p.badFrames([2 6]) = true;
    p.dynamicReference = false;
    p.corrMethod = 'fft';
    [~,p, position2, timeSec, rawPosition, peakValueArray] = ...
        StripAnalysis(videoArray, p); %#ok<*ASGLU>
   
    
    %% Third test
   
    % test the CUDA mode, iff there is a CUDA enabled GPU
    if gpuDeviceCount > 0
        p = struct;
        p.dynamicReference = true;
        p.minPeakThreshold = 0.45;
        p.corrMethod = 'cuda';
        p.enableVerbosity = 2;
        p.stripHeight = 15;
        p.stripWidth = 512;
        [~,p, position2, timeSec, rawPosition, peakValueArray] = ...
            StripAnalysis(videoArray, p); %#ok<*ASGLU>
    end
   
    success = true;
catch 
    success = false;
end

warning(origState);

