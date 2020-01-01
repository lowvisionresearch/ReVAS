function success = Tester_ReadSpecificFrame

% suppress warnings
origState = warning;
warning('off','all');

try
    %% read in sample video

    % the video resides under /testing folder.
    inputVideo = FindFile('tslo.avi');

    reader = VideoReader(inputVideo);
    frame = ReadSpecificFrame(reader, 15); %#ok<NASGU>
    
    success = true;
catch
    success = false;
end

warning(origState);