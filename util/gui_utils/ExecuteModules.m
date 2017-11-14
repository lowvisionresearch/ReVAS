function ExecuteModules(inputVideoPath, handles)
%EXECUTE MODULES Execute all enabled modules for one video.
%   This is a helper function for |JobQueue.m| that executes all enabled
%   modules for one video. Call this function in a loop to execute on all
%   videos.

addpath(genpath('..'));
global abortTriggered;
originalInputVideoPath = inputVideoPath(1:end-4);

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
if logical(handles.togTrim.Value) && ~logical(abortTriggered)
    RevasMessage(['Trimming ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.borderTrimAmount = handles.config.trimBorderTrimAmount;
    parametersStructure.overwrite = handles.config.trimOverwrite;

    % Call the function(s)
    TrimVideo(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_dwt' inputVideoPath(end-3:end)];
end

%% Remove Stimulus Module
if logical(handles.togStim.Value) && ~logical(abortTriggered)
    RevasMessage(['Removing Stimulus ' inputVideoPath], parametersStructure);
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

    % Call the function(s)
    if logical(handles.config.stimUseRectangle)
        FindStimulusLocations(inputVideoPath, stimulus, parametersStructure, ...
                              removalAreaSize);
    else
        FindStimulusLocations(inputVideoPath, stimulus, parametersStructure);
    end

    RemoveStimuli(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_nostim' inputVideoPath(end-3:end)];
end

%% Gamma Correction Module
if logical(handles.togGamma.Value) && ~logical(abortTriggered)
    RevasMessage(['Gamma Correcting ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.gammaExponent = handles.config.gammaExponent;
    parametersStructure.overwrite = handles.config.gammaOverwrite;

    % Call the function(s)
    GammaCorrect(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_gamscaled' inputVideoPath(end-3:end)];
end

%% Bandpass Filtering Module
if logical(handles.togBandFilt.Value) && ~logical(abortTriggered)
    RevasMessage(['Bandpass Filtering ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.smoothing = handles.config.bandFiltSmoothing;
    parametersStructure.lowSpatialFrequencyCutoff = handles.config.bandFiltFreqCut;
    parametersStructure.overwrite = handles.config.bandFiltOverwrite;

    % Call the function(s)
    BandpassFilter(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_bandfilt' inputVideoPath(end-3:end)];
end

%% Make Coarse Reference Frame Module
if (logical(handles.togCoarse.Value) ...
        || logical(handles.togFine.Value) ...
        || logical(handles.togStrip.Value)) && ~logical(abortTriggered)
    parametersStructure.overwrite = handles.config.coarseOverwrite || ...
        handles.config.fineOverwrite || ...
        handles.config.stripOverwrite;
    parametersStructure.thresholdValue = 4; % TODO (make changeable from GUI)
    parametersStructure.singleTail = true;
    parametersStructure.upperTail = false;
    FindBlinkFrames(inputVideoPath, parametersStructure);
end

if logical(handles.togCoarse.Value) && ~logical(abortTriggered)
    RevasMessage(['Making Coarse Reference Frame ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.refFrameNumber = handles.config.coarseRefFrameNum;
    parametersStructure.scalingFactor = handles.config.coarseScalingFactor;
    parametersStructure.overwrite = handles.config.coarseOverwrite;
    parametersStructure.enableVerbosity = handles.config.coarseVerbosity;
    parametersStructure.fileName = inputVideoPath;
    parametersStructure.adaptiveSearch = false;
    parametersStructure.enableSubpixelInterpolation = false;
    parametersStructure.enableGaussianFiltering = false;
    parametersStructure.enableGPU = false; % TODO
    % parametersStructure.maximumPeakRatio = Inf;
    parametersStructure.maximumPeakRatio = 0.35; % TODO make changeable from GUI, and check values
    % parametersStructure.minimumPeakThreshold = -Inf;
    parametersStructure.minimumPeakThreshold = 0.1; % TODO make changeable from GUI, and check values
    parametersStructure.axesHandles = [handles.axes1 handles.axes2 handles.axes3];

    % Call the function(s)
    coarseResult = CoarseRef(inputVideoPath, parametersStructure);

    % Update file name to output file name
    %localinputVideoPath = [localinputVideoPath(1:end-4) '_dwt' localinputVideoPath(end-3:end)];
end

%% Make Fine Reference Frame Module
if logical(handles.togFine.Value) && ~logical(abortTriggered)
    RevasMessage(['Making Fine Reference Frame ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.enableVerbosity = handles.config.fineVerbosity;
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

    % Call the function(s)
    fineResult = FineRef(coarseResult, inputVideoPath, parametersStructure);

    % Update file name to output file name
    %localinputVideoPath = [localinputVideoPath(1:end-4) '_dwt' localinputVideoPath(end-3:end)];
end

%% Strip Analysis Module
if logical(handles.togStrip.Value) && ~logical(abortTriggered)
    RevasMessage(['Strip Analyzing ' inputVideoPath], parametersStructure);
    % Set the parameters
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

    % Load a fine result if we didn't run the previous module in this
    % session.
    if ~exist('fineResult', 'var')
        if exist([originalInputVideoPath '_refframe.mat'], 'file')
           load([originalInputVideoPath '_refframe.mat']);
           fineResult = refFrame;
        else
            error('No reference frame available for strip analysis.');
        end
    end
    
    % Call the function(s)
    if strcmp(handles.togFine.Enable, 'on') 
        [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
            statisticsStructure] ...
            = StripAnalysis(inputVideoPath, fineResult, parametersStructure);
    elseif strcmp(handles.togCoarse.Enable, 'on')
        [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
            statisticsStructure] ...
            = StripAnalysis(inputVideoPath, coarseResult, parametersStructure);
    else
        % TODO use a specific frame of the video as reference
    end
    clear fineResult;
end

%% Re-referencing Module
if logical(handles.togReRef.Value) && ~logical(abortTriggered)
    RevasMessage(['Re-referencing ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.borderTrimAmount = handles.config.trimBorderTrimAmount;
    parametersStructure.overwrite = handles.config.trimOverwrite;

    % Call the function(s)
    TrimVideo(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_dwt' inputVideoPath(end-3:end)];
end

%% Filtering Module
if logical(handles.togFilt.Value) && ~logical(abortTriggered)
    RevasMessage(['Filtering ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.borderTrimAmount = handles.config.trimBorderTrimAmount;
    parametersStructure.overwrite = handles.config.trimOverwrite;

    % Call the function(s)
    TrimVideo(inputVideoPath, parametersStructure);

    % Update file name to output file name
    inputVideoPath = [inputVideoPath(1:end-4) '_dwt' inputVideoPath(end-3:end)];
end

%% Saccade Detection Module
if logical(handles.togSacDrift.Value) && ~logical(abortTriggered)
    RevasMessage(['Saccade Detecting ' inputVideoPath], parametersStructure);
    % Set the parameters
    parametersStructure.overwrite = handles.config.sacOverwrite;
    parametersStructure.enableVerbosity = handles.config.sacVerbosity;
    parametersStructure.thresholdValue = handles.config.sacThresholdVal;
    parametersStructure.secondaryThresholdValue = handles.config.sacSecThresholdVal;
    parametersStructure.stitchCriteria = handles.config.sacStitch;
    parametersStructure.minAmplitude = handles.config.sacMinAmplitude;
    parametersStructure.maxDuration = handles.config.sacMaxDuration;
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

    % Update file name to input file name
    inputFileName = [inputVideoPath(1:end-4) '_' ...
        int2str(parametersStructure.samplingRate) '_hz_final.mat'];

    % Call the function(s)
    % TODO
    FindSaccadesAndDrifts(inputFileName, [512 512], [10 10], ...
        parametersStructure);
end
end