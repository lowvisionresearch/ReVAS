function success = Tester_TrimVideo

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

% suppress warnings
origState = warning;
warning('off','all');

% use default params
p = struct; 
p.overwrite = true;
try
    % test with a video path
    outputVideoPath = TrimVideo(inputVideo, p); %#ok<*ASGLU>
    delete(outputVideoPath);
    
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3 5]) = true;
    outputVideo = TrimVideo(videoArray,p);
    if size(videoArray,3) - size(outputVideo,3) ~= sum(p.badFrames)
        success= false;
    else
        success = true;
    end
catch 
    success = false;
end

warning(origState);