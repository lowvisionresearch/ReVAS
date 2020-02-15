function success = Tester_RemoveStimuli

% suppress warnings
origState = warning;
warning('off','all');


try
    %% read in sample video

    % the video resides under /testing folder.
    inputVideo = FindFile('tslo-stim.avi');
    
    %% First test
    % use default params
    p = struct; 
    p.overwrite = true;
    p.removalAreaSize = [87 87];

    % test with a video path with default settings (11px white cross)
    [outputVideoPath, p, matFilePath, stimLocs] = RemoveStimuli(inputVideo, p); %#ok<*ASGLU>
    delete(matFilePath);
    
    %% Second test
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    
    % assume some of the frames were bad
    clear p;
    p = struct;
    p.overwrite = true;
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3 5]) = true;
    
    % stimulus is a large black cross
    p.stimulus = MakeStimulusCross(85, 19, 0); 
    
    [outputVideo, ~,~, stimLocs2] = RemoveStimuli(videoArray,p);
    
    % check if both methods give the same result. Note that 1 pixel shift
    % between white cross and stimulus is expected.
    assert(all(stimLocs(2,:)-1 == stimLocs2(2,:)));
    
    %% Third test
    p.enableVerbosity = 2;
    p.badFrames = false(11,1); % intentionally 2 frames more
    p.badFrames([2 6]) = true;
    p.fillingMethod = 'noise';
    [~, p3] = RemoveStimuli(videoArray,p);
    
    % check if results identical 
    assert(all(p3.stimulusLocations(2,:) == stimLocs2(2,:)));
    
    %% Fourth test
    % test with a video with stimulus already removed
    clear p;
    p = struct;
    p.overwrite = true;
    [newVideoPath, p, matFilePath, stimLocs4] = RemoveStimuli(outputVideoPath,p);
    delete(outputVideoPath);
    delete(newVideoPath);
    delete(matFilePath);
    
    assert(all(isnan(stimLocs4(:))));
    
    success = true;

catch 
    success = false;
end

warning(origState);

