function success = Tester_ReadSpecificFrame

% the video resides under /testing folder.
inputVideo = 'tslo.avi';

str = which(inputVideo);
if isempty(str)
    success = false;
    return;
else
    [filepath,name,ext] = fileparts(str); %#ok<*ASGLU>
    inputVideo = [filepath filesep inputVideo];
end

% suppress warnings
origState = warning;
warning('off','all');

try
    reader = VideoReader(inputVideo);
    frame = ReadSpecificFrame(reader, 15);
    
    success = true;
catch
    success = false;
end

warning(origState);