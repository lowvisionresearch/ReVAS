function success = Tester_ReReference

% suppress warnings
origState = warning;
warning('off','all');

try
    %% First test
    
    % use filepaths for both localRef and globalRef
    localRefPath = FindFile('tslo-localRef.mat');
    globalRefPath = FindFile('tslo-globalRef-tilted-3_25.tif');
    
    % use default params, but correct for torsion
    p = struct; 
    p.overwrite = true;
    p.fixTorsion = true;
    p.anchorStripHeight = 32;
    p.enableVerbosity = 1;
    p.globalRefArgument = globalRefPath;
    [offset, bestTilt, p, peakValues] = ReReference(localRefPath, p); %#ok<*ASGLU>
    delete(p.outputFilePath);
    
    assert(abs(bestTilt - 3.25) < 0.1);

    %% Second test
    
    % use preloaded arrays as localRef and globalRef. Rotate the localRef
    % and translate by a known amount and estimate the tilt and evaluate
    % re-referencing accuracy
    load(localRefPath,'params');
    localRef = params.referenceFrame;
    
    angle = -1.35;
    p.globalRefArgument = padarray(imrotate(localRef,angle,'bilinear'),21);
    [offset, bestTilt, p, peakValues] = ReReference(localRef, p);
    
    % check if we found the tilt correctly (to some degree)
    assert(abs(bestTilt - angle) < 0.1);

    success = true;

catch 
    success = false;
end

warning(origState);