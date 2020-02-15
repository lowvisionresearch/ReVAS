function success = Tester_ReReference

% suppress warnings
origState = warning;
warning('off','all');

try
    %% First test
    
    % use filepaths for both localRef and globalRef
    localRefPath = FindFile('tslo-localRef.mat');
    globalRefPath = FindFile('tslo-globalRef-tilted-3_25.tif');
    
    % make up some position traces
    position = rand(540,2);
    timeSec = linspace(0,1,540)';
    
    % use default params, but correct for torsion
    p = struct; 
    p.overwrite = true;
    p.fixTorsion = true;
    p.anchorStripHeight = 32;
    p.enableVerbosity = 1;
    p.globalRefArgument = globalRefPath;
    p.referenceFrame = localRefPath;
    [~,p,offset, bestTilt, peakValues] = ReReference([position timeSec], p); %#ok<*ASGLU>
    
    assert(abs(bestTilt - 3.25) < 0.1);

    %% Second test
    
    % use preloaded arrays as localRef and globalRef. Rotate the localRef
    % and translate by a known amount and estimate the tilt and evaluate
    % re-referencing accuracy
    load(localRefPath,'params');
    localRef = params.referenceFrame;
    
    % save made-up position traces to a hidden file
    hiddenFile = '.demoPositionFile.mat';
    save(hiddenFile,'position','timeSec');
    
    angle = -1.35;
    p.globalRefArgument = padarray(imrotate(localRef,angle,'bilinear'),21);
    [outputFilePath, p, offset, bestTilt, peakValues] = ReReference(hiddenFile, p);
    
    % check if we found the tilt correctly (to some degree)
    assert(abs(bestTilt - angle) < 0.1);
    
    delete(hiddenFile);
    delete(outputFilePath);

    success = true;

catch 
    success = false;
end

warning(origState);