function success = Tester_FindBlinkFrames

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample video
    
    % the video resides under /testing folder.
    inputVideo = FindFile('aoslo-blink.avi');   
    
    %% First test
    % use default params
    p = struct; 
    
    % test with a video path
    [~, p, outputFilePath, imStats, initialRef] = FindBlinkFrames(inputVideo, p); %#ok<*ASGLU>
    delete(outputFilePath);
    
    %% Second test
    % test with a video array
    p.enableVerbosity = true;
    videoArray = ReadVideoToArray(inputVideo);
    [~, p2, imStats, initialRef] = FindBlinkFrames(videoArray,p);

    assert(all(p.badFrames == p2.badFrames));
    
    %% Third test
    % video without a blink
    
    % the video resides under /testing folder.
    inputVideo = 'tslo-dark.avi';

    str = which(inputVideo);
    if isempty(str)
        success = false;
        return;
    else
        [filepath,name,ext] = fileparts(str);
        inputVideo = [filepath filesep inputVideo];
    end 
    % test with a video path
    [~, p3, outputFilePath, imStats, initialRef] = FindBlinkFrames(inputVideo, p); %#ok<*ASGLU>
    delete(outputFilePath);
    
    assert(sum(p3.badFrames) == 0);
    
    success = true;
catch
    success = false;
end

warning(origState);