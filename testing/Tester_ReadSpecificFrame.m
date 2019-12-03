function success = Tester_ReadSpecificFrame

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample video

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

    reader = VideoReader(inputVideo);
    frame = ReadSpecificFrame(reader, 15); %#ok<NASGU>
    
    success = true;
catch
    success = false;
end

warning(origState);