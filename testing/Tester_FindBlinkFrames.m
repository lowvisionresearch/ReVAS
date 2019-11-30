function success = Tester_FindBlinkFrames

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
try
    % test with a video path
    [badFrames, outputFilePath, imStats, initialRef] = FindBlinkFrames(inputVideo, p); %#ok<*ASGLU>
    delete(outputFilePath);
    
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    [badFrames, ~, imStats, initialRef] = FindBlinkFrames(videoArray, p);

    success = true;
catch 
    success = false;
end

warning(origState);