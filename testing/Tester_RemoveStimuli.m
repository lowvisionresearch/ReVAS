function success = Tester_RemoveStimuli

% the video resides under /testing folder.
inputVideo = 'tslo-stim.avi';

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
    p.removalAreaSize = 87;

    % test with a video path with default settings (11px white cross)
    [outputVideoPath, matFilePath, stimLocs] = RemoveStimuli(inputVideo, p); %#ok<*ASGLU>
    delete(outputVideoPath);
    delete(matFilePath);
    
    %% Second test
    % test with a video array
    videoArray = ReadVideoToArray(inputVideo);
    
    % assume some of the frames were bad
    clear p;
    p = struct;
    p.overwrite = true;
    p.badFrames = false(1,size(videoArray,3));
    p.badFrames([1 3 5]) = true;
    
    % stimulus is a large black cross
    p.stimulus = MakeStimulusCross(85, 19, 0); 
    
    [outputVideo, ~, stimLocs2] = RemoveStimuli(videoArray,p);
    
    % check if both methods give the same result. Note that 1 pixel shift
    % between white cross and stimulus is expected.
    assert(all(stimLocs(2,:)-1 == stimLocs2(2,:)));
    
    %% Third test
    p.enableVerbosity = true;
    p.badFrames = true(15,1); % intentionally out of bounds
    p.badFrames([2 4 6]) = false;
    p.fillingMethod = 'noise';
    [~, ~, stimLocs3] = RemoveStimuli(videoArray,p);
    
    % check if results identical 
    assert(all(stimLocs3(2,:) == stimLocs2(2,:)));
    
    success = true;

catch 
    success = false;
end

warning(origState);

