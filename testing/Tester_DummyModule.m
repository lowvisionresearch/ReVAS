function success = Tester_DummyModule

% suppress warnings
origState = warning;
warning('off','all');
success = true;

try
    
    %% read in sample video

    % the video resides under /testing folder.
    inputVideo = FindFile('aoslo.avi');
    
    
    %% First test
    % use default params
    p = struct; 
    p.overwrite = true;
    p.enableVerbosity = true;
    
    % test with a video path
    [outputVideoPath,p] = DummyModule(inputVideo, p); %#ok<*ASGLU>
    delete(p.matFilePath);
    
    %% Second test
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3 5]) = true;
    outputVideo = DummyModule(videoArray,p);
    
    % check if the difference in number of frames between two videos is
    % exactly equal to length of badFrames
    assert(size(videoArray,3) - size(outputVideo,3) == sum(p.badFrames));

    
    %% Third test
    % check if pixel value inversion works
    
    % read in inverted video, delete the file
    invertedVideoArray = ReadVideoToArray(p.outputVideoPath);
    delete(p.outputVideoPath);
    
    % check if sum of original video and inverted video is all 255
    assert(sum(sum((videoArray(:,:,2) + invertedVideoArray(:,:,2)) - uint8(255))) == 0);
    
    
catch
    success = false;
end

warning(origState);