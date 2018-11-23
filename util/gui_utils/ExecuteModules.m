function ExecuteModules(inputPath, handles)
%EXECUTE MODULES Execute all enabled modules for one video.
%   This is a helper function for |JobQueue.m| that executes all enabled
%   modules for one video. Call this function in a loop to execute on all
%   videos.

if ~isdeployed
    addpath(genpath('..'));
end

global abortTriggered;
originalInputVideoPath = inputPath(1:end-4);

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

parametersStructure = struct;
if ~logical(handles.config.parMultiCore) && ~logical(handles.config.parGPU)
    % Only print to GUI's "command window" if not parallelizing.
    parametersStructure.commandWindowHandle = handles.commandWindow;
end

% Set GPU option
if logical(handles.config.parGPU) && ~logical(abortTriggered)
    parametersStructure.enableGPU = true;
else
    parametersStructure.enableGPU = false;
end

%% Trim Module
if logical(handles.config.togValues('trim')) && ~logical(abortTriggered)
    RevasMessage(['[[ Trimming ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.borderTrimAmount = [
        handles.config.trimLeft, ...
        handles.config.trimRight, ...
        handles.config.trimTop, ...
        handles.config.trimBottom];
    parametersStructure.overwrite = handles.config.trimOverwrite;

    % Call the function
    TrimVideo(inputPath, parametersStructure);

    % Update file name to output file name
    inputPath = [inputPath(1:end-4) '_dwt' inputPath(end-3:end)];
end

%% Remove Stimulus Module
if logical(handles.config.togValues('stim')) && ~logical(abortTriggered)
    RevasMessage(['[[ Removing Stimulus ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.enableVerbosity = handles.config.stimVerbosity;
    parametersStructure.overwrite = handles.config.stimOverwrite;
    if logical(handles.config.stimOption1)
        stimulus = handles.config.stimFullPath;
    else
        stimulus = struct;
        stimulus.thickness = handles.config.stimThick;
        stimulus.size = handles.config.stimSize;
    end
    removalAreaSize = [handles.config.stimRectangleY, handles.config.stimRectangleX];
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];

    % Call the function
    if logical(handles.config.stimUseRectangle)
        RemoveStimuli(inputPath, stimulus, parametersStructure, ...
                              removalAreaSize);
    else
        RemoveStimuli(inputPath, stimulus, parametersStructure);
    end

    % Update file name to output file name
    inputPath = [inputPath(1:end-4) '_nostim' inputPath(end-3:end)];
end

%% Gamma Correction Module
if logical(handles.config.togValues('gamma')) && ~logical(abortTriggered)
    RevasMessage(['[[ Gamma Correcting ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.gammaExponent = handles.config.gammaExponent;
    parametersStructure.overwrite = handles.config.gammaOverwrite;

    % Call the function
    GammaCorrect(inputPath, parametersStructure);

    % Update file name to output file name
    inputPath = [inputPath(1:end-4) '_gamscaled' inputPath(end-3:end)];
end

%% Bandpass Filtering Module
if logical(handles.config.togValues('bandfilt')) && ~logical(abortTriggered)
    RevasMessage(['[[ Bandpass Filtering ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.smoothing = handles.config.bandFiltSmoothing;
    parametersStructure.lowSpatialFrequencyCutoff = handles.config.bandFiltFreqCut;
    parametersStructure.overwrite = handles.config.bandFiltOverwrite;

    % Call the function
    BandpassFilter(inputPath, parametersStructure);

    % Update file name to output file name
    inputPath = [inputPath(1:end-4) '_bandfilt' inputPath(end-3:end)];
end

%% Make Coarse Reference Frame Module
if (logical(handles.config.togValues('coarse')) ...
        || logical(handles.config.togValues('fine')) ...
        || logical(handles.config.togValues('strip'))) && ~logical(abortTriggered)
    RevasMessage(['Identifying blink frames in ' originalInputVideoPath], parametersStructure);
    parametersStructure.overwrite = handles.config.coarseOverwrite && ...
        handles.config.fineOverwrite && ...
        handles.config.stripOverwrite;
    parametersStructure.thresholdValue = 4; % TODO (make changeable from GUI)
    parametersStructure.singleTail = true;
    parametersStructure.upperTail = false;
    FindBlinkFrames(inputPath, parametersStructure);
end

if logical(handles.config.togValues('coarse')) && ~logical(abortTriggered)
    RevasMessage(['[[ Making Coarse Reference Frame ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.refFrameNumber = handles.config.coarseRefFrameNum;
    parametersStructure.scalingFactor = handles.config.coarseScalingFactor;
    parametersStructure.frameIncrement = handles.config.coarseFrameIncrement;
    parametersStructure.overwrite = handles.config.coarseOverwrite;
    parametersStructure.enableVerbosity = handles.config.coarseVerbosity;
    parametersStructure.fileName = inputPath;
    parametersStructure.adaptiveSearch = false;
    parametersStructure.enableSubpixelInterpolation = false;
    parametersStructure.enableGaussianFiltering = false;
    parametersStructure.enableGPU = false; % TODO
    % parametersStructure.maximumPeakRatio = Inf;
    parametersStructure.maximumPeakRatio = 0.8; % TODO make changeable from GUI, and check values
    % parametersStructure.minimumPeakThreshold = -Inf;
    parametersStructure.minimumPeakThreshold = 0.2; % TODO make changeable from GUI, and check values
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];

    % Call the function
    coarseRefFrame = CoarseRef(inputPath, parametersStructure);
end

%% Make Fine Reference Frame Module
if logical(handles.config.togValues('fine')) && ~logical(abortTriggered)
    RevasMessage(['[[ Making Fine Reference Frame ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.enableVerbosity = handles.config.fineVerbosity;
    parametersStructure.overwrite = handles.config.fineOverwrite;
    parametersStructure.numberOfIterations = handles.config.fineNumIterations;
    parametersStructure.stripHeight = handles.config.fineStripHeight;
    parametersStructure.stripWidth = handles.config.fineStripWidth;
    parametersStructure.samplingRate = handles.config.fineSamplingRate;
    parametersStructure.maximumPeakRatio = handles.config.fineMaxPeakRatio;
    parametersStructure.minimumPeakThreshold = handles.config.fineMinPeakThreshold;
    parametersStructure.adaptiveSearch = handles.config.fineAdaptiveSearch;
    parametersStructure.adaptiveSearchScalingFactor = handles.config.fineScalingFactor;
    parametersStructure.searchWindowHeight = handles.config.fineSearchWindowHeight;
    parametersStructure.enableSubpixelInterpolation = handles.config.fineSubpixelInterp;
    parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
        = handles.config.fineNeighborhoodSize;
    parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
        = handles.config.fineSubpixelDepth;
    parametersStructure.enableGaussianFiltering = false; % TODO
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];

    % Call the function
    [fineRefFrame, fineRefEyePositions, fineRefTimeArray] = ...
        FineRef(coarseRefFrame, inputPath, parametersStructure); %#ok<*ASGLU>
    eyePositionTraces = fineRefEyePositions;
    timeArray = fineRefTimeArray;
    save([inputPath(1:end-4) '_refframe.mat'],'eyePositionTraces','timeArray','-append');
end

%% Strip Analysis Module
if logical(handles.config.togValues('strip')) && ~logical(abortTriggered)
    RevasMessage(['[[ Strip Analyzing ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.createStabilizedVideo = handles.config.stripCreateStabilizedVideo;
    parametersStructure.overwrite = handles.config.stripOverwrite;
    parametersStructure.enableVerbosity = handles.config.stripVerbosity;
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];
    parametersStructure.commandWindowHandle = handles.commandWindow;    
    parametersStructure.stripHeight = handles.config.stripStripHeight;
    parametersStructure.stripWidth = handles.config.stripStripWidth;
    parametersStructure.samplingRate = handles.config.stripSamplingRate;
    parametersStructure.enableGaussianFiltering = handles.config.stripEnableGaussFilt;
    parametersStructure.gaussianStandardDeviation = handles.config.stripGaussSD;
    parametersStructure.maximumPeakRatio = handles.config.stripMaxPeakRatio;
    parametersStructure.minimumPeakThreshold = handles.config.stripMinPeakThreshold;
    parametersStructure.adaptiveSearch = handles.config.stripAdaptiveSearch;
    parametersStructure.adaptiveSearchScalingFactor = handles.config.stripScalingFactor;
    parametersStructure.searchWindowHeight = handles.config.stripSearchWindowHeight;
    parametersStructure.enableSubpixelInterpolation = ...
        handles.config.stripSubpixelInterp;
    parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
        = handles.config.stripNeighborhoodSize;
    parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
        = handles.config.stripSubpixelDepth;
    parametersStructure.maximumSD = handles.config.stripGaussSD;
    parametersStructure.SDWindowSize = handles.config.stripSDWindow;

    % Load a fine ref if we didn't run the previous module in this
    % session.
    if ~exist('fineRefFrame', 'var')
        if ~isempty(strfind(originalInputVideoPath, '_dwt')) %#ok<*STREMP>
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_dwt')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_nostim'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_nostim')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_gamscaled'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_gamscaled')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_bandfilt'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_bandfilt')-1);
        else
            rawVideoPath = originalInputVideoPath;
        end
            
        % Try to identify a fine ref to use from file.
        fineRefFrames = dir([rawVideoPath '*refframe.mat']);
        % Try to identify a coarse ref to use from file.
        coarseRefFrames = dir([rawVideoPath '*coarseref.mat']);
        
        if logical(handles.config.togValues('coarse'))
           % Use coarse ref frame if fine ref module disabled and if
           % available.
           fineRefFrame = coarseRefFrame;
        elseif ~isempty({fineRefFrames.name})
           % Load a saved fine ref if available.
           % If there are multiple such files, we make the assumption that
           % we desire the one with longest file name since that had the
           % most amount of processing applied to it.
           longestFileName = '';
           for name = {fineRefFrames.name}
               if length(name) > length(longestFileName)
                   longestFileName = name{1};
               end
           end
           RevasMessage(['Loading fine reference frame from: ' longestFileName], parametersStructure);
           load(longestFileName, 'refFrame');
           fineRefFrame = refFrame;
        elseif ~isempty({coarseRefFrames.name})
           % Load a saved coarse ref if available.
           % If there are multiple such files, we make the assumption that
           % we desire the one with longest file name since that had the
           % most amount of processing applied to it.
           longestFileName = '';
           for name = {coarseRefFrames.name}
               if length(name) > length(longestFileName)
                   longestFileName = name{1};
               end
           end
           RevasMessage(['Loading coarse reference frame from: ' longestFileName], parametersStructure);
           load(longestFileName, 'refFrame');
           fineRefFrame = refFrame;
        else
           RevasError(originalInputVideoPath, ...
               'No reference frame available for strip analysis.', ...
               parametersStructure);
           return;
        end
    end
    
    % Call the function
    StripAnalysis(inputPath, fineRefFrame, parametersStructure);
    
    % Update file name to input file name
    inputPath = [inputPath(1:end-4) '_' ...
        int2str(parametersStructure.samplingRate) '_hz_final.mat'];
    
    % Write output file as csv
    load(inputPath, ...
        'timeArray', ...
        'eyePositionTraces', ...
        'rawEyePositionTraces', ...
        'peakRatios');
    csvPath = [inputPath(1:end-4) '.csv'];
    fid = fopen(csvPath, 'w');
    fprintf(fid, '%s\n', cell2mat({ ...
        'time,' ...
        'eyePositionTraceHoriz,' ...
        'eyePositionTraceVert,' ...
        'rawEyePositionTraceHoriz,' ...
        'rawEyePositionTraceVert,' ...
        'peakRatio,' ...
        }));
    fclose(fid);
    dlmwrite(csvPath, [ ...
        timeArray, ...
        eyePositionTraces, ...
        rawEyePositionTraces, ...
        peakRatios], '-append');
end

%% Re-referencing Module
if logical(handles.config.togValues('reref')) && strcmp(handles.config.rerefGlobalFullPath, '')
    RevasMessage(['[[ Re-referencing ]] ' inputPath], parametersStructure);
    RevasMessage('No valid global reference frame provided, skipping Re-Referencing', parametersStructure);
    
elseif logical(handles.config.togValues('reref')) && ~logical(abortTriggered)
    RevasMessage(['[[ Re-referencing ]] ' inputPath], parametersStructure);
    % Set the parameters    
    parametersStructure.verbosity = handles.config.rerefVerbosity;
    parametersStructure.overwrite = handles.config.rerefOverwrite;
    parametersStructure.searchZone = handles.config.rerefSearch;
    parametersStructure.findPeakMethod = handles.config.rerefPeakMethod;
    if handles.config.rerefPeakMethod == 2
        parametersStructure.findPeakKernelSize = handles.config.rerefKernel;
    end
    parametersStructure.fixTorsion = handles.config.rerefTorsion;
    if handles.config.rerefTorsion
        parametersStructure.tilts = handles.config.rerefTiltLow: ...
            handles.config.rerefTiltStep: ...
            handles.config.rerefTiltUp;
    end
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];
    globalRefFrame = handles.config.rerefGlobalFullPath;
    
    % Load a fine ref if we didn't run the previous module in this
    % session.
    if ~exist('fineRefFrame', 'var')
        if ~isempty(strfind(originalInputVideoPath, '_dwt'))%JG replaced ~isempty(strfind(
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_dwt')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_nostim'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_nostim')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_gamscaled'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_gamscaled')-1);
        elseif ~isempty(strfind(originalInputVideoPath, '_bandfilt'))
            rawVideoPath = originalInputVideoPath(1:strfind(originalInputVideoPath, '_bandfilt')-1);
        else
            rawVideoPath = originalInputVideoPath;
        end
            
        % Try to identify a fine ref to use from file.
        fineRefFrames = dir([rawVideoPath '*refframe.mat']);
        % Try to identify a coarse ref to use from file.
        coarseRefFrames = dir([rawVideoPath '*coarseref.mat']);
        
        if logical(handles.config.togValues('coarse'))
           % Use coarse ref frame if fine ref module disabled and if
           % available.
           fineRefFrame = coarseRefFrame;
        elseif ~isempty({fineRefFrames.name})
           % Load a saved fine ref if available.
           % If there are multiple such files, we make the assumption that
           % we desire the one with longest file name since that had the
           % most amount of processing applied to it.
           longestFileName = '';
           for name = {fineRefFrames.name}
               if length(name) > length(longestFileName)
                   longestFileName = name{1};
               end
           end
           RevasMessage(['Loading fine reference frame from: ' longestFileName], parametersStructure);
           load(fullfile(fileparts(rawVideoPath),longestFileName), 'refFrame');
           fineRefFrame = refFrame;
        elseif ~isempty({coarseRefFrames.name})
           % Load a saved coarse ref if available.
           % If there are multiple such files, we make the assumption that
           % we desire the one with longest file name since that had the
           % most amount of processing applied to it.
           longestFileName = '';
           for name = {coarseRefFrames.name}
               if length(name) > length(longestFileName)
                   longestFileName = name{1};
               end
           end
           RevasMessage(['Loading coarse reference frame from: ' longestFileName], parametersStructure);
           load(fullfile(fileparts(rawVideoPath),longestFileName), 'refFrame');
           fineRefFrame = refFrame;
        else
           RevasError(originalInputVideoPath, ...
               'No reference frame available for re-referencing.', ...
               parametersStructure);
           return;
        end
    end
    
    
    % if strip analysis is skipped, only fine ref is done, then use the eye
    % position traces from refframe.mat file
    if ~logical(handles.config.togValues('strip'))
        inputPath = [inputPath(1:end-4) '_refframe.mat'];
    end
    
    % Call the function
    [~,inputPath] = ReReference(inputPath, fineRefFrame, globalRefFrame, ...
        parametersStructure);
    
    % Write output file as csv
    load(inputPath, ...
        'timeArray', ...
        'eyePositionTraces');
    csvPath = [inputPath(1:end-4) '.csv'];
    fid = fopen(csvPath, 'w');
    fprintf(fid, '%s\n', cell2mat({ ...
        'time,' ...
        'eyePositionTraceHoriz,' ...
        'eyePositionTraceVert,' ...
        }));
    fclose(fid);
    dlmwrite(csvPath, [ ...
        timeArray, ...
        eyePositionTraces], '-append');
end

%% Filtering Module
if logical(handles.config.togValues('filt')) && ~logical(abortTriggered)
    RevasMessage(['[[ Filtering ]] ' inputPath], parametersStructure);
    % Set the parameters    
    parametersStructure.overwrite = handles.config.filtOverwrite;
    parametersStructure.verbosity = handles.config.filtVerbosity;
    parametersStructure.FirstPrefilter = handles.config.filtFirstPrefilter;
    parametersStructure.SecondPrefilter = handles.config.filtSecondPrefilter;
    parametersStructure.maxGapDurationMs = handles.config.filtMaxGapDur;
    parametersStructure.filterTypes = {};
    parametersStructure.filterParameters = {};
    
    if handles.config.filtEnableMedian1
        % Median Filtering
        parametersStructure.filterTypes{1} = @medfilt1;
        parametersStructure.filterParameters{1} = handles.config.filtMedian1;
    else
        % SGO Lay Filtering
        parametersStructure.filterTypes{1} = @sgolayfilt;
        parametersStructure.filterParameters{1} = [handles.config.filtPoly1, ...
            handles.config.filtKernel1];
    end
    if handles.config.filtEnableMedian2
        % Median Filtering
        parametersStructure.filterTypes{2} = @medfilt1;
        parametersStructure.filterParameters{2} = handles.config.filtMedian2;
    elseif handles.config.filtEnableSgo2
        % SGO Lay Filtering
        parametersStructure.filterTypes{2} = @sgolayfilt;
        parametersStructure.filterParameters{2} = [handles.config.filtPoly2, ...
            handles.config.filtKernel2];
    end
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];
    
    % Call the function
    [~,inputPath] = FilterEyePosition(inputPath, parametersStructure);
    
    % Write output file as csv
    load(inputPath, ...
        'timeArray', ...
        'eyePositionTraces');
    csvPath = [inputPath(1:end-4) '.csv'];
    fid = fopen(csvPath, 'w');
    fprintf(fid, '%s\n', cell2mat({ ...
        'time,' ...
        'eyePositionTraceHoriz,' ...
        'eyePositionTraceVert,' ...
        }));
    fclose(fid);
    dlmwrite(csvPath, [ ...
        timeArray, ...
        eyePositionTraces], '-append');
end

%% Saccade Detection Module
if logical(handles.config.togValues('sacdrift')) && ~logical(abortTriggered)
    RevasMessage(['[[ Saccade Detecting ]] ' inputPath], parametersStructure);
    % Set the parameters
    parametersStructure.overwrite = handles.config.sacOverwrite;
    parametersStructure.enableVerbosity = handles.config.sacVerbosity;
    parametersStructure.isAdaptive = handles.config.sacIsAdaptive;
    parametersStructure.isMedianBased = handles.config.sacIsMedianBased;
    parametersStructure.pixelSize = handles.config.sacPixelSize;
    parametersStructure.thresholdValue = handles.config.sacThresholdVal;
    parametersStructure.secondaryThresholdValue = handles.config.sacSecThresholdVal;
    parametersStructure.stitchCriteria = handles.config.sacStitch;
    parametersStructure.minAmplitude = handles.config.sacMinAmplitude;
    parametersStructure.maxDuration = handles.config.sacMaxDuration;
    parametersStructure.minDuration = handles.config.sacMinDuration;
    if handles.config.sacDetectionMethod1
        parametersStructure.detectionMethod = 1;
    else
        parametersStructure.detectionMethod = 2;
    end
    parametersStructure.hardVelocityThreshold = handles.config.sacHardVelThreshold;
    parametersStructure.hardSecondaryVelocityThreshold = ...
        handles.config.sacHardSecondaryVelThreshold;
    if handles.config.sacVelMethod1
        parametersStructure.velocityMethod = 1;
    else
        parametersStructure.velocityMethod = 2;
    end
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];

    % Call the function
    [inputPath,~,~] = FindSaccadesAndDrifts(inputPath,parametersStructure);
    
    % Write output file as csv
    load(inputPath, ...
        'saccades', ...
        'drifts');
    csvPath = [inputPath(1:end-4) '.csv'];
    fid = fopen(csvPath, 'w');
    headerRow = fieldnames(saccades); %#ok<*NODEF>
    headerRow(1:end-1) = strcat(headerRow(1:end-1), ',');
    headerRow = [{'saccadeOrDrift,'}; headerRow];
    fprintf(fid, '%s\n', cell2mat(headerRow'));
    saccades = cell2mat(struct2cell(saccades))';
    drifts = cell2mat(struct2cell(drifts))';
    for i = 1:size(saccades, 1)
        fprintf(fid, '%s', 'saccade,');
        fprintf(fid, '%f,', saccades(i,:));
        fprintf(fid, '\n');
    end
    for i = 1:size(drifts, 1)
        fprintf(fid, '%s', 'drift,');
        fprintf(fid, '%f,', drifts(i,:));
        fprintf(fid, '\n');
    end
    fclose(fid);
end
end