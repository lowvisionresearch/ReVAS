function success = Tester_BandpassFilter

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

% suppress warnings
origState = warning;
warning('off','all');


try
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
    p.badFrames = true(15,1); % intentionally out of bounds
    p.badFrames([2 4 6]) = false;
    outputVideo2 = BandpassFilter(videoArray,p);
    
    assert(size(outputVideo2,3) == size(videoArray,3));
    
    success = true;
    
    %% cleanup
    delete(outputVideoPath);

catch 
    success = false;
end

warning(origState);

