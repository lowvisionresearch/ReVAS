function success = Tester_TrimVideo

% suppress warnings
origState = warning;
warning('off','all');

try
    
    %% read in sample video

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
    
    
    %% First test
    % use default params
    p = struct; 
    p.overwrite = true;
    
    % test with a video path
    outputVideoPath = TrimVideo(inputVideo, p); %#ok<*ASGLU>
    delete(outputVideoPath);
    
    %% Second test
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    p.borderTrimAmount = [11 20 30 41];
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3 5]) = true;
    outputVideo = TrimVideo(videoArray,p);
    
    % check if the difference in number of frames between two videos is
    % exactly equal to length of badFrames
    if size(videoArray,3) - size(outputVideo,3) ~= sum(p.badFrames)
        success= false;
    else
        success = true;
    end
    
catch 
    success = false;
end

warning(origState);