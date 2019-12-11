function success = Tester_FindBlinkFrames

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample video
    
    % the video resides under /testing folder.
    inputVideo = 'aoslo-blink.avi';

    str = which(inputVideo);
    if isempty(str)
        success = false;
        return;
    else
        [filepath,name,ext] = fileparts(str);
        inputVideo = [filepath filesep inputVideo];
    end       
    
    %% First test
    % use default params
    p = struct; 
    
    % test with a video path
    [badFrames, outputFilePath, imStats, initialRef] = FindBlinkFrames(inputVideo, p); %#ok<*ASGLU>
    delete(outputFilePath);
    
    %% Second test
    % test with a video array
    p.enableVerbosity = true;
    videoArray = ReadVideoToArray(inputVideo);
    [badFrames2, ~, imStats, initialRef,params] = FindBlinkFrames(videoArray,p);

    assert(all(badFrames2 == badFrames));
    
    %% Third test
    
    success = true;
catch 
    success = false;
end

warning(origState);