function varargout = JobQueue(varargin)
% JOBQUEUE MATLAB code for JobQueue.fig
%      JOBQUEUE, by itself, creates a new JOBQUEUE or raises the existing
%      singleton*.
%
%      H = JOBQUEUE returns the handle to a new JOBQUEUE or the handle to
%      the existing singleton*.
%
%      JOBQUEUE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in JOBQUEUE.M with the given input arguments.
%
%      JOBQUEUE('Property','Value',...) creates a new JOBQUEUE or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before JobQueue_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to JobQueue_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help JobQueue

% Last Modified by GUIDE v2.5 03-Jul-2017 15:06:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @JobQueue_OpeningFcn, ...
                   'gui_OutputFcn',  @JobQueue_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before JobQueue is made visible.
function JobQueue_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to JobQueue (see VARARGIN)

% Choose default command line output for JobQueue
handles.output = hObject;

% COLOR PALETTE
% http://paletton.com/palette.php?uid=c491l5-2L0kj0tK00%2B%2B6SNBlToaqM2g
handles.colors = ... % primary, white, light, dark, black
    {[114,102,173],[255,255,255],[206,201,226],[ 67, 53,133],[  5,  3, 12];
     [229, 93,106],[255,255,255],[251,198,203],[186, 59, 71],[ 17,  3,  4];
     [101,198, 80],[255,255,255],[195,237,187],[ 70,161, 51],[  5, 15,  2];
     [237,209, 96],[255,255,255],[255,244,200],[209,180, 69],[ 18, 15,  3]};

for i = 1:size(handles.colors,1)
    for j = 1:size(handles.colors,2)
        handles.colors{i,j} = handles.colors{i,j}/255;
    end
end
 
% Set colors
% Main Background
handles.jobQueue.Color = handles.colors{1,2};
% Box backgrounds
handles.inputVideoBox.BackgroundColor = handles.colors{1,3};
handles.radioRaw.BackgroundColor = handles.colors{1,3};
handles.radioTrim.BackgroundColor = handles.colors{1,3};
handles.radioNoStim.BackgroundColor = handles.colors{1,3};
handles.radioGamma.BackgroundColor = handles.colors{1,3};
handles.radioBandFilt.BackgroundColor = handles.colors{1,3};
handles.modulesBox.BackgroundColor = handles.colors{1,3};
handles.textTrim.BackgroundColor = handles.colors{1,3};
handles.textStim.BackgroundColor = handles.colors{1,3};
handles.textGamma.BackgroundColor = handles.colors{1,3};
handles.textBandFilt.BackgroundColor = handles.colors{1,3};
handles.textCoarse.BackgroundColor = handles.colors{1,3};
handles.textFine.BackgroundColor = handles.colors{1,3};
handles.textStrip.BackgroundColor = handles.colors{1,3};
handles.textFilt.BackgroundColor = handles.colors{1,3};
handles.textReRef.BackgroundColor = handles.colors{1,3};
handles.textSacDrift.BackgroundColor = handles.colors{1,3};
% Box text
handles.inputVideoBox.ForegroundColor = handles.colors{1,5};
handles.radioRaw.ForegroundColor = handles.colors{1,5};
handles.radioTrim.ForegroundColor = handles.colors{1,5};
handles.radioNoStim.ForegroundColor = handles.colors{1,5};
handles.radioGamma.ForegroundColor = handles.colors{1,5};
handles.radioBandFilt.ForegroundColor = handles.colors{1,5};
handles.modulesBox.ForegroundColor = handles.colors{1,5};
handles.textTrim.ForegroundColor = handles.colors{1,5};
handles.textStim.ForegroundColor = handles.colors{1,5};
handles.textGamma.ForegroundColor = handles.colors{1,5};
handles.textBandFilt.ForegroundColor = handles.colors{1,5};
handles.textCoarse.ForegroundColor = handles.colors{1,5};
handles.textFine.ForegroundColor = handles.colors{1,5};
handles.textStrip.ForegroundColor = handles.colors{1,5};
handles.textFilt.ForegroundColor = handles.colors{1,5};
handles.textReRef.ForegroundColor = handles.colors{1,5};
handles.textSacDrift.ForegroundColor = handles.colors{1,5};
% Select/Enable buttons backgrounds
handles.selectFiles.BackgroundColor = handles.colors{1,4};
handles.togTrim.BackgroundColor = handles.colors{1,4};
handles.togStim.BackgroundColor = handles.colors{1,4};
handles.togGamma.BackgroundColor = handles.colors{1,4};
handles.togBandFilt.BackgroundColor = handles.colors{1,4};
handles.togCoarse.BackgroundColor = handles.colors{1,4};
handles.togFine.BackgroundColor = handles.colors{1,4};
handles.togStrip.BackgroundColor = handles.colors{1,4};
handles.togFilt.BackgroundColor = handles.colors{1,4};
handles.togReRef.BackgroundColor = handles.colors{1,4};
handles.togSacDrift.BackgroundColor = handles.colors{1,4};
% Select/Enable button text
handles.selectFiles.ForegroundColor = handles.colors{1,2};
handles.togTrim.ForegroundColor = handles.colors{1,2};
handles.togStim.ForegroundColor = handles.colors{1,2};
handles.togGamma.ForegroundColor = handles.colors{1,2};
handles.togBandFilt.ForegroundColor = handles.colors{1,2};
handles.togCoarse.ForegroundColor = handles.colors{1,2};
handles.togFine.ForegroundColor = handles.colors{1,2};
handles.togStrip.ForegroundColor = handles.colors{1,2};
handles.togFilt.ForegroundColor = handles.colors{1,2};
handles.togReRef.ForegroundColor = handles.colors{1,2};
handles.togSacDrift.ForegroundColor = handles.colors{1,2};
% Configure buttons backgrounds
handles.configTrim.BackgroundColor = handles.colors{4,4};
handles.configStim.BackgroundColor = handles.colors{4,4};
handles.configGamma.BackgroundColor = handles.colors{4,4};
handles.configBandFilt.BackgroundColor = handles.colors{4,4};
handles.configCoarse.BackgroundColor = handles.colors{4,4};
handles.configFine.BackgroundColor = handles.colors{4,4};
handles.configStrip.BackgroundColor = handles.colors{4,4};
handles.configFilt.BackgroundColor = handles.colors{4,4};
handles.configReRef.BackgroundColor = handles.colors{4,4};
handles.configSacDrift.BackgroundColor = handles.colors{4,4};
% Configure button text
handles.configTrim.ForegroundColor = handles.colors{4,2};
handles.configStim.ForegroundColor = handles.colors{4,2};
handles.configGamma.ForegroundColor = handles.colors{4,2};
handles.configBandFilt.ForegroundColor = handles.colors{4,2};
handles.configCoarse.ForegroundColor = handles.colors{4,2};
handles.configFine.ForegroundColor = handles.colors{4,2};
handles.configStrip.ForegroundColor = handles.colors{4,2};
handles.configFilt.ForegroundColor = handles.colors{4,2};
handles.configReRef.ForegroundColor = handles.colors{4,2};
handles.configSacDrift.ForegroundColor = handles.colors{4,2};
% Parallelization button background
handles.parallelization.BackgroundColor = handles.colors{4,4};
% Parallelization button text
handles.parallelization.ForegroundColor = handles.colors{4,2};
% Execute button background
handles.execute.BackgroundColor = handles.colors{3,4};
% Execute button text
handles.execute.ForegroundColor = handles.colors{3,2};

% DEFAULT PARAMETERS
% Trim
handles.config.trimBorderTrimAmount = 24;
handles.config.trimOverwrite = true;
% Stim
handles.config.stimVerbosity = true;
handles.config.stimOverwrite = true;
% Gamma
handles.config.gammaExponent = 0.6;
handles.config.gammaOverwrite = true;
% BandFilt
handles.config.bandFiltSmoothing = 1;
handles.config.bandFiltFreqCut = 3.0;
handles.config.bandFiltOverwrite = true;
% Coarse
handles.config.coarseRefFrameNum = 15;
handles.config.coarseScalingFactor = 0.5;
handles.config.coarseOverwrite = true;
handles.config.coarseVerbosity = true;
% Fine
handles.config.fineOverwrite = true;
handles.config.fineVerbosity = true;
handles.config.fineNumIterations = 1;
handles.config.fineStripHeight = 15;
handles.config.fineStripWidth = 488;
handles.config.fineSamplingRate = 540;
handles.config.fineMinPeakRatio = 0.8;
handles.config.fineMinPeakThreshold = 0.2;
handles.config.fineAdaptiveSearch = false;
handles.config.fineScalingFactor = 8;
handles.config.fineSearchWindowHeight = 79;
handles.config.fineSubpixelInterp = true;
handles.config.fineNeighborhoodSize = 7;
handles.config.fineSubpixelDepth = 2;
% Strip
handles.config.stripOverwrite = true;
handles.config.stripVerbosity = true;
handles.config.stripStripHeight = 15;
handles.config.stripStripWidth = 488;
handles.config.stripSamplingRate = 540;
handles.config.stripEnableGaussFilt = true;
handles.config.stripDisableGaussFilt = false;
handles.config.stripGaussSD = 10;
handles.config.stripMinPeakRatio = 0.8;
handles.config.stripMinPeakThreshold = 0;
handles.config.stripAdaptiveSearch = false;
handles.config.stripScalingFactor = 8;
handles.config.stripSearchWindowHeight = 79;
handles.config.stripSubpixelInterp = true;
handles.config.stripNeighborhoodSize = 7;
handles.config.stripSubpixelDepth = 2;
% Sac
handles.config.sacOverwrite = true;
handles.config.sacVerbosity = true;
handles.config.sacThresholdVal = 6;
handles.config.sacSecThresholdVal = 2;
handles.config.sacStitch = 15;
handles.config.sacMinAmplitude = 0.1;
handles.config.sacMaxDuration = 100;
handles.config.sacDetectionMethod1 = false;
handles.config.sacHardVelThreshold = 35;
handles.config.sacHardSecondaryVelThreshold = 35;
handles.config.sacDetectionMethod2 = true;
handles.config.sacVelMethod1 = true;
handles.config.sacVelMethod2 = false;
% Parallelization
handles.config.parMultiCore = false;
handles.config.parGPU = false;

% Pre-Disabled Toggle Values
handles.config.preDisabledTogTrimValue = 1;
handles.config.preDisabledTogStimValue = 1;
handles.config.preDisabledTogGammaValue = 1;
handles.config.preDisabledTogBandFiltValue = 1;

% Pre-Disabled Execute Screen GUI Items
handles.execute1.Visible = 'off';
handles.execute2.Visible = 'off';
handles.abort.Visible = 'off';

% Initial files
handles.files = cell(0);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes JobQueue wait for user response (see UIRESUME)
% uiwait(handles.jobQueue);


% --- Outputs from this function are returned to the command line.
function varargout = JobQueue_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in parallelization.
function parallelization_Callback(hObject, eventdata, handles)
% hObject    handle to parallelization (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Parallelization;

% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
% hObject    handle to reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

initialize_gui(gcbf, handles, true);

% --------------------------------------------------------------------


% --- Executes on button press in add.
function add_Callback(hObject, eventdata, handles)
% hObject    handle to add (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Open the Add Module GUI
AddModule;

% --- Executes on button press in execute.
function execute_Callback(hObject, eventdata, handles)
% hObject    handle to execute (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Update visible and invisible gui components
handles.inputVideoBox.Visible = 'off';
handles.selectFiles.Visible = 'off';
handles.inputList.Visible = 'off';
handles.modulesBox.Visible = 'off';
handles.textTrim.Visible = 'off';
handles.textStim.Visible = 'off';
handles.textGamma.Visible = 'off';
handles.textBandFilt.Visible = 'off';
handles.textCoarse.Visible = 'off';
handles.textFine.Visible = 'off';
handles.textStrip.Visible = 'off';
handles.textFilt.Visible = 'off';
handles.textReRef.Visible = 'off';
handles.textSacDrift.Visible = 'off';
handles.togTrim.Visible = 'off';
handles.togStim.Visible = 'off';
handles.togGamma.Visible = 'off';
handles.togBandFilt.Visible = 'off';
handles.togCoarse.Visible = 'off';
handles.togFine.Visible = 'off';
handles.togStrip.Visible = 'off';
handles.togFilt.Visible = 'off';
handles.togReRef.Visible = 'off';
handles.togSacDrift.Visible = 'off';
handles.configTrim.Visible = 'off';
handles.configStim.Visible = 'off';
handles.configGamma.Visible = 'off';
handles.configBandFilt.Visible = 'off';
handles.configCoarse.Visible = 'off';
handles.configFine.Visible = 'off';
handles.configStrip.Visible = 'off';
handles.configFilt.Visible = 'off';
handles.configReRef.Visible = 'off';
handles.configSacDrift.Visible = 'off';
handles.parallelization.Visible = 'off';
handles.execute.Visible = 'off';

handles.execute1.Visible = 'on';
handles.execute2.Visible = 'on';
handles.abort.Visible = 'on';

drawnow;

% Setup parfor loop for multi-core processing
if logical(handles.config.parMultiCore)
    localCluster = parcluster('local');
    numberOfWorkers = localCluster.NumWorkers;
else
    numberOfWorkers = 0;
end

% Apply modules to all selected files
parfor (i = 1:size(handles.files, 2), numberOfWorkers)
    localHandles = handles;
    parametersStructure = struct;
    % Set GPU option
    if logical(localHandles.config.parGPU)
        parametersStructure.enableGPU = true;
    else
        parametersStructure.enableGPU = false;
    end
    
    if logical(localHandles.togTrim.Value)
        % Set the parameters
        parametersStructure.borderTrimAmount = localHandles.config.trimBorderTrimAmount;
        parametersStructure.overwrite = localHandles.config.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_dwt' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togStim.Value)
        % Set the parameters
        parametersStructure.enableVerbosity = localHandles.config.stimVerbosity;
        parametersStructure.overwrite = localHandles.config.stimOverwrite;
        stimulus = struct;
        stimulus.thickness = 1; % TODO
        stimulus.size = 51; % TODO
        parametersStructure.thresholdValue = 4; % TODO (used for blink detection)
        
        % Call the function(s)
        FindBlinkFrames(localHandles.files{i}, parametersStructure);
        FindStimulusLocations(localHandles.files{i}, stimulus, parametersStructure);
        RemoveStimuli(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_nostim' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togGamma.Value)
        % Set the parameters
        parametersStructure.gammaExponent = localHandles.config.gammaExponent;
        parametersStructure.overwrite = localHandles.config.gammaOverwrite;
        
        % Call the function(s)
        GammaCorrect(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_gamscaled' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togBandFilt.Value)
        % Set the parameters
        parametersStructure.smoothing = localHandles.config.bandFiltSmoothing;
        parametersStructure.lowSpatialFrequencyCutoff = localHandles.config.bandFiltFreqCut;
        parametersStructure.overwrite = localHandles.config.bandFiltOverwrite;
        
        % Call the function(s)
        BandpassFilter(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_bandfilt' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togCoarse.Value) % TODO
        % Set the parameters
        parametersStructure.refFrameNumber = localHandles.config.coarseRefFrameNum;
        parametersStructure.scalingFactor = localHandles.config.coarseScalingFactor;
        parametersStructure.overwrite = localHandles.config.coarseOverwrite;
        parametersStructure.enableVerbosity = localHandles.config.coarseVerbosity;
        parametersStructure.fileName = localHandles.files{i};
        parametersStructure.enableGPU = false; % TODO

        % Call the function(s)
        coarseResult = CoarseRef(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        %localHandles.files{i} = [localHandles.files{i}(1:end-4) '_dwt' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togFine.Value)
        % Set the parameters
        parametersStructure.enableVerbosity = localHandles.config.fineVerbosity;
        parametersStructure.numberOfIterations = localHandles.config.fineNumIterations;
        parametersStructure.stripHeight = localHandles.config.fineStripHeight;
        parametersStructure.stripWidth = localHandles.config.fineStripWidth;
        parametersStructure.samplingRate = localHandles.config.fineSamplingRate;
        parametersStructure.minimumPeakRatio = localHandles.config.fineMinPeakRatio;
        parametersStructure.minimumPeakThreshold = localHandles.config.fineMinPeakThreshold;
        parametersStructure.adaptiveSearch = localHandles.config.fineAdaptiveSearch;
        parametersStructure.adaptiveSearchScalingFactor = localHandles.config.fineScalingFactor;
        parametersStructure.searchWindowHeight = localHandles.config.fineSearchWindowHeight;
        parametersStructure.enableSubpixelInterpolation = localHandles.config.fineSubpixelInterp;
        parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
            = localHandles.config.fineNeighborhoodSize;
        parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
            = localHandles.config.fineSubpixelDepth;
        parametersStructure.enableGaussianFiltering = false; % TODO
        parametersStructure.badFrames = []; % TODO
        parametersStructure.axeslocalHandles = []; % TODO        
        
        % Call the function(s)
        fineResult = FineRef(coarseResult, localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        %localHandles.files{i} = [localHandles.files{i}(1:end-4) '_dwt' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togStrip.Value)
        % Set the parameters
        parametersStructure.overwrite = localHandles.config.stripOverwrite;
        parametersStructure.enableVerbosity = localHandles.config.stripVerbosity;
        parametersStructure.stripHeight = localHandles.config.stripStripHeight;
        parametersStructure.stripWidth = localHandles.config.stripStripWidth;
        parametersStructure.samplingRate = localHandles.config.stripSamplingRate;
        parametersStructure.enableGaussianFiltering = localHandles.config.stripEnableGaussFilt;
        parametersStructure.gaussianStandardDeviation = localHandles.config.stripGaussSD;
        parametersStructure.minimumPeakRatio = localHandles.config.stripMinPeakRatio;
        parametersStructure.minimumPeakThreshold = localHandles.config.stripMinPeakThreshold;
        parametersStructure.adaptiveSearch = localHandles.config.stripAdaptiveSearch;
        parametersStructure.adaptiveSearchScalingFactor = localHandles.config.stripScalingFactor;
        parametersStructure.searchWindowHeight = localHandles.config.stripSearchWindowHeight;
        parametersStructure.enableSubpixelInterpolation = ...
            localHandles.config.stripSubpixelInterp;
        parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
            = localHandles.config.stripNeighborhoodSize;
        parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
            = localHandles.config.stripSubpixelDepth;

        % Call the function(s)
        if strcmp(localHandles.togFine.Enable, 'on') 
            [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
                statisticsStructure] ...
                = StripAnalysis(localHandles.files{i}, fineResult, parametersStructure);
        elseif strcmp(localHandles.togCoarse.Enable, 'on')
            [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
                statisticsStructure] ...
                = StripAnalysis(localHandles.files{i}, coarseResult, parametersStructure);
        else
            % TODO use a specific frame of the video as reference
        end        
    end
    
    if false
    %if logical(localHandles.togFilt.Value) % TODO
        % Set the parameters
        parametersStructure.borderTrimAmount = localHandles.config.trimBorderTrimAmount;
        parametersStructure.overwrite = localHandles.config.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_dwt' localHandles.files{i}(end-3:end)];
    end
    
    if false
    %if logical(localHandles.togReRef.Value) % TODO
        % Set the parameters
        parametersStructure.borderTrimAmount = localHandles.config.trimBorderTrimAmount;
        parametersStructure.overwrite = localHandles.config.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(localHandles.files{i}, parametersStructure);
        
        % Update file name to output file name
        localHandles.files{i} = [localHandles.files{i}(1:end-4) '_dwt' localHandles.files{i}(end-3:end)];
    end
    
    if logical(localHandles.togSacDrift.Value)
        % Set the parameters
        parametersStructure.overwrite = localHandles.config.sacOverwrite;
        parametersStructure.enableVerbosity = localHandles.config.sacVerbosity;
        parametersStructure.thresholdValue = localHandles.config.sacThresholdVal;
        parametersStructure.secondaryThresholdValue = localHandles.config.sacSecThresholdVal;
        parametersStructure.stitchCriteria = localHandles.config.sacStitch;
        parametersStructure.minAmplitude = localHandles.config.sacMinAmplitude;
        parametersStructure.maxDuration = localHandles.config.sacMaxDuration;
        if localHandles.config.sacDetectionMethod1
            parametersStructure.detectionMethod = 1;
        else
            parametersStructure.detectionMethod = 2;
        end
        parametersStructure.hardVelocityThreshold = localHandles.config.sacHardVelThreshold;
        parametersStructure.hardSecondaryVelocityThreshold = ...
            localHandles.config.sacHardSecondaryVelThreshold;
        if localHandles.config.sacVelMethod1
            parametersStructure.velocityMethod = 1;
        else
            parametersStructure.velocityMethod = 2;
        end
        
        % Update file name to input file name
        inputFileName = [localHandles.files{i}(1:end-4) '_' ...
            int2str(parametersStructure.samplingRate) '_hz_final.mat'];

        % Call the function(s)
        % TODO
        FindSaccadesAndDrifts(inputFileName, [512 512], [10 10], ...
            parametersStructure);
    end
end

fprintf('Process Completed\n');


% --- Executes on button press in radioRaw.
function radioRaw_Callback(hObject, eventdata, handles)
% hObject    handle to radioRaw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioRaw
if strcmp(handles.togTrim.Enable, 'off')
    handles.togTrim.Value = handles.config.preDisabledTogTrimValue;
    handles.togTrim.Enable = 'on';
    handles.configTrim.Enable = 'on';
    togTrim_Callback(handles.togTrim, eventdata, handles);
end

if strcmp(handles.togStim.Enable, 'off')
    handles.togStim.Value = handles.config.preDisabledTogStimValue;
    handles.togStim.Enable = 'on';
    handles.configStim.Enable = 'on';
    togStim_Callback(handles.togStim, eventdata, handles);
end

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.config.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.config.preDisabledTogBandFiltValue;
    handles.togBandFilt.Enable = 'on';
    handles.configBandFilt.Enable = 'on';
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in selectFiles.
function selectFiles_Callback(hObject, eventdata, handles)
% hObject    handle to selectFiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.radioRaw, 'Value')
    suffix = '.avi';
elseif get(handles.radioTrim, 'Value')
    suffix = '_dwt.avi';        
elseif get(handles.radioNoStim, 'Value')
    suffix = '_nostim.avi';    
elseif get(handles.radioGamma, 'Value')
    suffix = '_gamscaled.avi';    
elseif get(handles.radioBandFilt, 'Value')
    suffix = '_bandfilt.avi'; 
else
    suffix = '';
end

handles.files = uipickfiles('FilterSpec', '*.avi');

% Go through list of selected items and filter
i = 1;
while i <= size(handles.files, 2)
    if ~iscell(handles.files) && handles.files == 0
        % User canceled file selection
        return;
    elseif isdir(handles.files{i})
        % Pull out any files contained within a selected folder
        % Save the path
        folder = handles.files{i};
        % Delete the path from the list
        handles.files(i) = [];
        
        % Append files to our list of files if they match the suffix
        folderFiles = dir(folder);
        for j = i:size(folderFiles, 1)
            if contains(folderFiles(j).name, suffix) && ...
                    (~strcmp('.avi', suffix) || ...
                    isempty(findstr('_dwt', folderFiles(j).name)) && ...
                    isempty(findstr('_nostim', folderFiles(j).name)) && ...
                    isempty(findstr('_gamscaled', folderFiles(j).name)) && ...
                    isempty(findstr('_bandfilt', folderFiles(j).name)))
                handles.files = ...
                    [{fullfile(folderFiles(j).folder, folderFiles(j).name)}, ...
                    handles.files];
            end
        end
    elseif ~contains(handles.files{i}, suffix) || ...
            (strcmp('.avi', suffix) && ...
                    (~isempty(findstr('_dwt', handles.files{i})) || ...
                    ~isempty(findstr('_nostim', handles.files{i})) || ...
                    ~isempty(findstr('_gamscaled', handles.files{i})) || ...
                    ~isempty(findstr('_bandfilt', handles.files{i}))))
        % Purge list of any items not matching the selected input video
        % type
        handles.files(i) = [];
    else
        % Only increment if we did not delete an item this iteration.
        i = i + 1;
    end
end

% Display final list of files back in the gui
handles.files = sort(handles.files);
displayFileList = handles.files;
for i = 1:size(handles.files, 2)
    [~,displayFileList{i},~] = fileparts(handles.files{i});
    displayFileList{i} = [displayFileList{i} '.avi'];
end
handles.inputList.String = displayFileList';

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioBandFilt.
function radioBandFilt_Callback(hObject, eventdata, handles)
% hObject    handle to radioBandFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioBandFilt
if strcmp(handles.togTrim.Enable, 'on')
    handles.config.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.config.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'on')
    handles.config.preDisabledTogGammaValue = handles.togGamma.Value;
end
handles.togGamma.Value = 0;
handles.togGamma.Enable = 'off';
handles.configGamma.Enable = 'off';
togGamma_Callback(handles.togGamma, eventdata, handles);

if strcmp(handles.togBandFilt.Enable, 'on')
    handles.config.preDisabledTogBandFiltValue = handles.togBandFilt.Value;
end
handles.togBandFilt.Value = 0;
handles.togBandFilt.Enable = 'off';
handles.configBandFilt.Enable = 'off';
togBandFilt_Callback(handles.togBandFilt, eventdata, handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes when selected cell(s) is changed in inputList.
function inputList_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to inputList (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in inputList.
function inputList_Callback(hObject, eventdata, handles)
% hObject    handle to inputList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns inputList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from inputList


% --- Executes during object creation, after setting all properties.
function inputList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to inputList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: inputList controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in togTrim.
function togTrim_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togStrip.
function togStrip_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    if handles.togFilt.Value == 1 || ...
            handles.togReRef.Value == 1 || ...
            handles.togSacDrift.Value == 1
        errordlg(...
            'Strip Analysis must be enabled if Filtering, Re-Referencing, or Saccade Detection is enabled.', 'Invalid Selection');
        hObject.Value = 1;
        return;
    end
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togStim.
function togStim_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togGamma.
function togGamma_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togCoarse.
function togCoarse_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    if handles.togFine.Value == 1
        errordlg(...
            'Make Coarse Reference Frame must be enabled if Make Fine Reference Frame is enabled.', 'Invalid Selection');
        hObject.Value = 1;
        return;
    end
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togReRef.
function togReRef_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    if handles.togStrip.Value == 0
        warndlg('Strip Analysis has been enabled since it must be if Re-Referencing is enabled.', 'Input Warning');
        handles.togStrip.Value = 1;
        togStrip_Callback(handles.togStrip, eventdata, handles);
    end
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togFilt.
function togFilt_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    if handles.togStrip.Value == 0
        warndlg('Strip Analysis has been enabled since it must be if Filtering is enabled.', 'Input Warning');
        handles.togStrip.Value = 1;
        togStrip_Callback(handles.togStrip, eventdata, handles);
    end
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togSacDrift.
function togSacDrift_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    if handles.togStrip.Value == 0
        warndlg('Strip Analysis has been enabled since it must be if Filtering is enabled.', 'Input Warning');
        handles.togStrip.Value = 1;
        togStrip_Callback(handles.togStrip, eventdata, handles);
    end
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togFine.
function togFine_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    if handles.togCoarse.Value == 0
        warndlg('Make Coarse Reference Frame has been enabled since it must be if Make Fine Reference Frame is enabled.', 'Input Warning');
        handles.togCoarse.Value = 1;
        togCoarse_Callback(handles.togCoarse, eventdata, handles);
    end
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in togBandFilt.
function togBandFilt_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
end

% --- Executes on button press in configTrim.
function configTrim_Callback(hObject, eventdata, handles)
% hObject    handle to configTrim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
TrimParameters;

% --- Executes on button press in configStim.
function configStim_Callback(hObject, eventdata, handles)
% hObject    handle to configStim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
StimParameters;

% --- Executes on button press in configGamma.
function configGamma_Callback(hObject, eventdata, handles)
% hObject    handle to configGamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
GammaParameters;

% --- Executes on button press in configBandFilt.
function configBandFilt_Callback(hObject, eventdata, handles)
% hObject    handle to configBandFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
BandFiltParameters;

% --- Executes on button press in configFine.
function configFine_Callback(hObject, eventdata, handles)
% hObject    handle to configFine (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FineParameters;

% --- Executes on button press in configCoarse.
function configCoarse_Callback(hObject, eventdata, handles)
% hObject    handle to configCoarse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
CoarseParameters;

% --- Executes on button press in configStrip.
function configStrip_Callback(hObject, eventdata, handles)
% hObject    handle to configStrip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
StripParameters;

% --- Executes on button press in configFilt.
function configFilt_Callback(hObject, eventdata, handles)
% hObject    handle to configFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in configReRef.
function configReRef_Callback(hObject, eventdata, handles)
% hObject    handle to configReRef (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in configSacDrift.
function configSacDrift_Callback(hObject, eventdata, handles)
% hObject    handle to configSacDrift (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
SacDriftParameters;


% --- Executes on button press in radioTrim.
function radioTrim_Callback(hObject, eventdata, handles)
% hObject    handle to radioTrim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioTrim
if strcmp(handles.togTrim.Enable, 'on')
    handles.config.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'off')
    handles.togStim.Value = handles.config.preDisabledTogStimValue;
    handles.togStim.Enable = 'on';
    handles.configStim.Enable = 'on';
    togStim_Callback(handles.togStim, eventdata, handles);
end

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.config.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.config.preDisabledTogBandFiltValue;
    handles.togBandFilt.Enable = 'on';
    handles.configBandFilt.Enable = 'on';
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
end

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in radioNoStim.
function radioNoStim_Callback(hObject, eventdata, handles)
% hObject    handle to radioNoStim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioNoStim
if strcmp(handles.togTrim.Enable, 'on')
    handles.config.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.config.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.config.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.config.preDisabledTogBandFiltValue;
    handles.togBandFilt.Enable = 'on';
    handles.configBandFilt.Enable = 'on';
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioGamma.
function radioGamma_Callback(hObject, eventdata, handles)
% hObject    handle to radioGamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioGamma
if strcmp(handles.togTrim.Enable, 'on')
    handles.config.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.config.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'on')
    handles.config.preDisabledTogGammaValue = handles.togGamma.Value;
end
handles.togGamma.Value = 0;
handles.togGamma.Enable = 'off';
handles.configGamma.Enable = 'off';
togGamma_Callback(handles.togGamma, eventdata, handles);

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.config.preDisabledTogBandFiltValue;
    handles.togBandFilt.Enable = 'on';
    handles.configBandFilt.Enable = 'on';
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
end

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in abort.
function abort_Callback(hObject, eventdata, handles)
% hObject    handle to abort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuAbout_Callback(hObject, eventdata, handles)
% hObject    handle to menuAbout (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
msgbox({'Retinal Video Analysis Suite (ReVAS)'; ...
    'Copyright (c) August 2017.'; ...
    'Sight Enhancement Laboratory at Berkeley.'; ...
    'School of Optometry.'; ...
    'University of California, Berkeley, USA.'; ...
    ''; ...
    'Mehmet N. Agaoglu, PhD.'; ...
    'mna@berkeley.edu.'; ...
    ''; ...
    'Matthew T. Sit.'; ...
    'msit@berkeley.edu.'; ...
    ''; ...
    'Derek Wan.'; ...
    'derek.wan11@berkeley.edu.'; ...
    ''; ...
    'Susana T. L. Chung, OD, PhD.'; ...
    's.chung@berkeley.edu.'}, ...
    'About ReVAS');

% --------------------------------------------------------------------
function menuLoad_Callback(hObject, eventdata, handles)
% hObject    handle to menuLoad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName, pathName, ~] = uigetfile('*.mat', 'Load Configurations','configurations.mat');
if fileName == 0
    % User canceled.
    return;
end
configurationsStruct = struct;
toggleButtonStates = [];

% Suppress warning if variables not found in loaded file
warning('off','MATLAB:load:variableNotFound');

load(fullfile(pathName, fileName), 'configurationsStruct', 'toggleButtonStates');

% Copying over data (values not provided will remain unchanged)
configurationsStructFieldNames = fieldnames(configurationsStruct);
for i = 1:length(configurationsStructFieldNames)
    handles.config.(configurationsStructFieldNames{i}) = ...
        configurationsStruct.(configurationsStructFieldNames{i});
end
if size(toggleButtonStates,1) == 1 && size(toggleButtonStates,2) == 10
    handles.togTrim.Value = toggleButtonStates(1);
    handles.togStim.Value = toggleButtonStates(2);
    handles.togGamma.Value = toggleButtonStates(3);
    handles.togBandFilt.Value = toggleButtonStates(4);
    handles.togCoarse.Value = toggleButtonStates(5);
    handles.togFine.Value = toggleButtonStates(6);
    handles.togStrip.Value = toggleButtonStates(7);
    handles.togFilt.Value = toggleButtonStates(8);
    handles.togReRef.Value = toggleButtonStates(9);
    handles.togSacDrift.Value = toggleButtonStates(10);
    
    togTrim_Callback(handles.togTrim, eventdata, handles);
    togStim_Callback(handles.togStim, eventdata, handles);
    togGamma_Callback(handles.togGamma, eventdata, handles);
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
    togFine_Callback(handles.togFine, eventdata, handles);
    togStrip_Callback(handles.togStrip, eventdata, handles);
    togFilt_Callback(handles.togFilt, eventdata, handles);
    togReRef_Callback(handles.togReRef, eventdata, handles);
    togSacDrift_Callback(handles.togSacDrift, eventdata, handles);
end

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function menuSave_Callback(hObject, eventdata, handles)
% hObject    handle to menuSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName,pathName, ~] = uiputfile('*.mat','Save Configurations','configurations.mat');
if fileName == 0
    % User canceled.
    return;
end
configurationsStruct = handles.config; %#ok<NASGU>
toggleButtonStates = [handles.togTrim.Value ...
    handles.togStim.Value ...
    handles.togGamma.Value ...
    handles.togBandFilt.Value ...
    handles.togCoarse.Value ...
    handles.togFine.Value ...
    handles.togStrip.Value ...
    handles.togFilt.Value ...
    handles.togReRef.Value ...
    handles.togSacDrift.Value];
if strcmp(handles.togTrim.Enable, 'off')
    toggleButtonStates(1) = handles.config.preDisabledTogTrimValue;
end
if strcmp(handles.togStim.Enable, 'off')
    toggleButtonStates(2) = handles.config.preDisabledTogStimValue;
end
if strcmp(handles.togGamma.Enable, 'off')
    toggleButtonStates(3) = handles.config.preDisabledTogGammaValue;
end
if strcmp(handles.togBandFilt.Enable, 'off')
    toggleButtonStates(4) = handles.config.preDisabledTogBandFiltValue; %#ok<NASGU>
end
save(fullfile(pathName, fileName), 'configurationsStruct', 'toggleButtonStates');

% --------------------------------------------------------------------
function menuExit_Callback(hObject, eventdata, handles)
% hObject    handle to menuExit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;