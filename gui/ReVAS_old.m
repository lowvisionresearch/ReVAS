function varargout = ReVAS(varargin)
% REVAS MATLAB code for ReVAS.fig
%      REVAS, by itself, creates a new REVAS or raises the existing
%      singleton*.
%
%      H = REVAS returns the handle to a new REVAS or the handle to
%      the existing singleton*.
%
%      REVAS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REVAS.M with the given input arguments.
%
%      REVAS('Property','Value',...) creates a new REVAS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Main_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Main_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ReVAS

% Last Modified by GUIDE v2.5 26-Jul-2018 01:38:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Main_OpeningFcn, ...
                   'gui_OutputFcn',  @Main_OutputFcn, ...
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

% --- Executes just before ReVAS is made visible.
function Main_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ReVAS (see VARARGIN)

% clear the command window
clc;

% Choose default command line output for ReVAS
handles.output = hObject;

set(hObject,'CloseRequestFcn',@revasCloseRequest);

if ~isdeployed
    addpath(genpath('..'));
end

try
    p = which('ReVAS');
    load([fileparts(p) filesep 'uiconfig.mat'],'GUIposition','uiFontSize',...
        'uiTitleFontSize','revasColors');
catch 
    disp('''uiconfig.mat'' file cannot be loaded. Executing UIConfigMaker.m')
    disp('to create a configuration file with default values.')
    disp('Using UIConfigMaker.m to make a new configuration file.')
    [GUIposition, uiFontSize,...
        uiTitleFontSize,revasColors] = UIConfigMaker;
end

% update the handles
handles.GUIposition = GUIposition;
handles.uiFontSize = uiFontSize;
handles.uiTitleFontSize = uiTitleFontSize;
handles.revasColors = revasColors;

% set a proper size for the main GUI window. a
handles.revas.Units = 'normalized';
handles.revas.OuterPosition = handles.GUIposition.revas;

% Update handles structure
guidata(hObject, handles);

% set font size and size and position of the GUI
InitGUIHelper(handles, handles.revas);
 
% Set colors
% ReVAS Background
handles.revas.Color = revasColors.background;
handles.inputList.BackgroundColor = revasColors.background;
handles.axes1.XColor = revasColors.background;
handles.axes1.YColor = revasColors.background;
handles.axes2.XColor = revasColors.background;
handles.axes2.YColor = revasColors.background;
handles.axes3.XColor = revasColors.background;
handles.axes3.YColor = revasColors.background;
handles.commandWindow.BackgroundColor = revasColors.background;

% Box backgrounds
handles.inputVideoBox.BackgroundColor = revasColors.boxBackground;
handles.radioRaw.BackgroundColor = revasColors.boxBackground;
handles.radioTrim.BackgroundColor = revasColors.boxBackground;
handles.radioNoStim.BackgroundColor = revasColors.boxBackground;
handles.radioGamma.BackgroundColor = revasColors.boxBackground;
handles.radioBandFilt.BackgroundColor = revasColors.boxBackground;
handles.radioStrip.BackgroundColor = revasColors.boxBackground;
handles.modulesBox.BackgroundColor = revasColors.boxBackground;
handles.textTrim.BackgroundColor = revasColors.boxBackground;
handles.textStim.BackgroundColor = revasColors.boxBackground;
handles.textGamma.BackgroundColor = revasColors.boxBackground;
handles.textBandFilt.BackgroundColor = revasColors.boxBackground;
handles.textCoarse.BackgroundColor = revasColors.boxBackground;
handles.textFine.BackgroundColor = revasColors.boxBackground;
handles.textStrip.BackgroundColor = revasColors.boxBackground;
handles.textReRef.BackgroundColor = revasColors.boxBackground;
handles.textFilt.BackgroundColor = revasColors.boxBackground;
handles.textSacDrift.BackgroundColor = revasColors.boxBackground;
% Box text
handles.inputList.ForegroundColor = revasColors.text;
handles.inputVideoBox.ForegroundColor = revasColors.text;
handles.radioRaw.ForegroundColor = revasColors.text;
handles.radioTrim.ForegroundColor = revasColors.text;
handles.radioNoStim.ForegroundColor = revasColors.text;
handles.radioGamma.ForegroundColor = revasColors.text;
handles.radioBandFilt.ForegroundColor = revasColors.text;
handles.radioStrip.ForegroundColor = revasColors.text;
handles.modulesBox.ForegroundColor = revasColors.text;
handles.textTrim.ForegroundColor = revasColors.text;
handles.textStim.ForegroundColor = revasColors.text;
handles.textGamma.ForegroundColor = revasColors.text;
handles.textBandFilt.ForegroundColor = revasColors.text;
handles.textCoarse.ForegroundColor = revasColors.text;
handles.textFine.ForegroundColor = revasColors.text;
handles.textStrip.ForegroundColor = revasColors.text;
handles.textReRef.ForegroundColor = revasColors.text;
handles.textFilt.ForegroundColor = revasColors.text;
handles.textSacDrift.ForegroundColor = revasColors.text;
handles.commandWindow.ForegroundColor = revasColors.text;
% Select/Enable buttons backgrounds
handles.selectFiles.BackgroundColor = revasColors.pushButtonBackground;
handles.togTrim.BackgroundColor = revasColors.activeButtonBackground;
handles.togStim.BackgroundColor = revasColors.activeButtonBackground;
handles.togGamma.BackgroundColor = revasColors.activeButtonBackground;
handles.togBandFilt.BackgroundColor = revasColors.activeButtonBackground;
handles.togCoarse.BackgroundColor = revasColors.activeButtonBackground;
handles.togFine.BackgroundColor = revasColors.activeButtonBackground;
handles.togStrip.BackgroundColor = revasColors.activeButtonBackground;
handles.togReRef.BackgroundColor = revasColors.activeButtonBackground;
handles.togFilt.BackgroundColor = revasColors.activeButtonBackground;
handles.togSacDrift.BackgroundColor = revasColors.activeButtonBackground;
% Select/Enable button text
handles.selectFiles.ForegroundColor = revasColors.pushButtonText;
handles.togTrim.ForegroundColor = revasColors.activeButtonText;
handles.togStim.ForegroundColor = revasColors.activeButtonText;
handles.togGamma.ForegroundColor = revasColors.activeButtonText;
handles.togBandFilt.ForegroundColor = revasColors.activeButtonText;
handles.togCoarse.ForegroundColor = revasColors.activeButtonText;
handles.togFine.ForegroundColor = revasColors.activeButtonText;
handles.togStrip.ForegroundColor = revasColors.activeButtonText;
handles.togReRef.ForegroundColor = revasColors.activeButtonText;
handles.togFilt.ForegroundColor = revasColors.activeButtonText;
handles.togSacDrift.ForegroundColor = revasColors.activeButtonText;
% Configure buttons backgrounds
handles.configTrim.BackgroundColor = revasColors.pushButtonBackground;
handles.configStim.BackgroundColor = revasColors.pushButtonBackground;
handles.configGamma.BackgroundColor = revasColors.pushButtonBackground;
handles.configBandFilt.BackgroundColor = revasColors.pushButtonBackground;
handles.configCoarse.BackgroundColor = revasColors.pushButtonBackground;
handles.configFine.BackgroundColor = revasColors.pushButtonBackground;
handles.configStrip.BackgroundColor = revasColors.pushButtonBackground;
handles.configReRef.BackgroundColor = revasColors.pushButtonBackground;
handles.configFilt.BackgroundColor = revasColors.pushButtonBackground;
handles.configSacDrift.BackgroundColor = revasColors.pushButtonBackground;
% Configure button text
handles.configTrim.ForegroundColor = revasColors.pushButtonText;
handles.configStim.ForegroundColor = revasColors.pushButtonText;
handles.configGamma.ForegroundColor = revasColors.pushButtonText;
handles.configBandFilt.ForegroundColor = revasColors.pushButtonText;
handles.configCoarse.ForegroundColor = revasColors.pushButtonText;
handles.configFine.ForegroundColor = revasColors.pushButtonText;
handles.configStrip.ForegroundColor = revasColors.pushButtonText;
handles.configReRef.ForegroundColor = revasColors.pushButtonText;
handles.configFilt.ForegroundColor = revasColors.pushButtonText;
handles.configSacDrift.ForegroundColor = revasColors.pushButtonText;
% Parallelization button background
handles.parallelization.BackgroundColor = revasColors.pushButtonBackground;
% Parallelization button text
handles.parallelization.ForegroundColor = revasColors.pushButtonText;
% Execute button background
handles.execute.BackgroundColor = revasColors.pushButtonBackground;
% Execute button text
handles.execute.ForegroundColor = revasColors.pushButtonText;
% Save Log button background
handles.saveLog.BackgroundColor = revasColors.pushButtonBackground;
% Save Log button text
handles.saveLog.ForegroundColor = revasColors.pushButtonText;
% Re-Config button background
handles.reconfig.BackgroundColor = revasColors.pushButtonBackground;
% Re-Config button text
handles.reconfig.ForegroundColor = revasColors.pushButtonText;

% DEFAULT PARAMETERS
% Trim
handles.config.trimLeft = 0;
handles.config.trimRight = 0;
handles.config.trimTop = 12;
handles.config.trimBottom = 0;
handles.config.trimOverwrite = true;
% Stim
handles.config.stimVerbosity = true;
handles.config.stimOverwrite = true;
handles.config.stimOption1 = false;
handles.config.stimOption2 = true;
handles.config.stimPath = '';
handles.config.stimFullPath = '';
handles.config.stimSize = 11;
handles.config.stimThick = 1;
handles.config.stimRectangleX = 11;
handles.config.stimRectangleY = 11;
handles.config.stimUseRectangle = false;
% Gamma
handles.config.gammaExponent = 0.6;
handles.config.gammaOverwrite = true;
handles.config.isGammaCorrect = true;
handles.config.isHistEq = false;
% BandFilt
handles.config.bandFiltSmoothing = 1;
handles.config.bandFiltFreqCut = 3.0;
handles.config.bandFiltOverwrite = true;
% Coarse
handles.config.coarseRefFrameNum = 3;
handles.config.coarseScalingFactor = 1.0;
handles.config.coarseFrameIncrement = 1;
handles.config.coarseOverwrite = true;
handles.config.coarseVerbosity = true;
% Fine
handles.config.fineOverwrite = true;
handles.config.fineVerbosity = true;
handles.config.fineNumIterations = 1;
handles.config.fineStripHeight = 11;
handles.config.fineStripWidth = 512;
handles.config.fineSamplingRate = 540;
handles.config.fineMaxPeakRatio = 0.65;
handles.config.fineMinPeakThreshold = 0.3;
handles.config.fineAdaptiveSearch = true;
handles.config.fineScalingFactor = 1;
handles.config.fineSearchWindowHeight = 79;
handles.config.fineSubpixelInterp = false;
handles.config.fineNeighborhoodSize = 7;
handles.config.fineSubpixelDepth = 2;
% Strip
handles.config.stripCreateStabilizedVideo = false;
handles.config.stripOverwrite = true;
handles.config.stripVerbosity = true;
handles.config.stripStripHeight = 15;
handles.config.stripStripWidth = 512;
handles.config.stripSamplingRate = 540;
handles.config.stripEnableGaussFilt = false;
handles.config.stripDisableGaussFilt = true;
handles.config.stripGaussSD = 10;
handles.config.stripSDWindow = 25;
handles.config.stripMaxPeakRatio = 0.8;
handles.config.stripMinPeakThreshold = 0.2;
handles.config.stripAdaptiveSearch = true;
handles.config.stripScalingFactor = 1;
handles.config.stripSearchWindowHeight = 79;
handles.config.stripSubpixelInterp = false;
handles.config.stripNeighborhoodSize = 7;
handles.config.stripSubpixelDepth = 2;
% Re-Referencing
handles.config.rerefOverwrite = true;
handles.config.rerefVerbosity = true;
handles.config.rerefSearch = 0.5;
handles.config.rerefPeakMethod = 2;
handles.config.rerefKernel = 21;
handles.config.rerefTorsion = true;
handles.config.rerefTiltLow = -5;
handles.config.rerefTiltUp = 5;
handles.config.rerefTiltStep = 1;
handles.config.rerefGlobalPath = '';
handles.config.rerefGlobalFullPath = '';
% Filtering
handles.config.filtFirstPrefilter = false;
handles.config.filtSecondPrefilter = false;
handles.config.filtOverwrite = true;
handles.config.filtVerbosity = true;
handles.config.filtMaxGapDur = 10;
handles.config.filtEnableMedian1 = true;
handles.config.filtEnableSgo1 = false;
handles.config.filtMedian1 = 11;
handles.config.filtPoly1 = 3;
handles.config.filtKernel1 = 15;
handles.config.filtEnableMedian2 = false;
handles.config.filtEnableSgo2 = true;
handles.config.filtEnableNoFilt2 = false;
handles.config.filtMedian2 = 11;
handles.config.filtPoly2 = 3;
handles.config.filtKernel2 = 15;
% Sac
handles.config.sacOverwrite = true;
handles.config.sacVerbosity = true;
handles.config.sacIsAdaptive = true;
handles.config.sacIsMedianBased = false;
handles.config.sacPixelSize = 10*60/512;
handles.config.sacThresholdVal = 6;
handles.config.sacSecThresholdVal = 3;
handles.config.sacStitch = 15;
handles.config.sacMinAmplitude = 0.1;
handles.config.sacMaxDuration = 100;
handles.config.sacMinDuration = 8;
handles.config.sacDetectionMethod1 = false;
handles.config.sacHardVelThreshold = 25;
handles.config.sacHardSecondaryVelThreshold = 15;
handles.config.sacDetectionMethod2 = true;
handles.config.sacVelMethod1 = false;
handles.config.sacVelMethod2 = true;
% Parallelization
handles.config.parMultiCore = false;
handles.config.parGPU = false;

handles.moduleNames = {
    'trim', ...
    'stim', ...
    'gamma', ...
    'bandfilt', ...
    'coarse', ...
    'fine', ...
    'strip', ...
    'reref', ...
    'filt', ...
    'sacdrift'};

% For toggle buttons
% Local lists of references, to be converted to maps below.
togHandleRefs = {
    handles.togTrim, ...
    handles.togStim, ...
    handles.togGamma, ...
    handles.togBandFilt, ...
    handles.togCoarse, ...
    handles.togFine, ...
    handles.togStrip, ...
    handles.togReRef, ...
    handles.togFilt, ...
    handles.togSacDrift};
configHandleRefs = {
    handles.configTrim, ...
    handles.configStim, ...
    handles.configGamma, ...
    handles.configBandFilt, ...
    handles.configCoarse, ...
    handles.configFine, ...
    handles.configStrip, ...
    handles.configReRef, ...
    handles.configFilt, ...
    handles.configSacDrift};
togCallbackRefs = {
    @togTrim_Callback, ...
    @togStim_Callback, ...
    @togGamma_Callback, ...
    @togBandFilt_Callback, ...
    @togCoarse_Callback, ...
    @togFine_Callback, ...
    @togStrip_Callback, ...
    @togReRef_Callback, ...
    @togFilt_Callback, ...
    @togSacDrift_Callback};



% Used when push buttons are to be disabled by the radio buttons, in
% which case the should temporarily show as disabled.
handles.config.preDisabledTogValues = ...
    containers.Map(handles.moduleNames(1:7), num2cell(zeros(1,7)));
% Handles to each toggle button.
handles.togHandles = ...
    containers.Map(handles.moduleNames, togHandleRefs);
% Handles to each config button.
handles.configHandles = ...
    containers.Map(handles.moduleNames, configHandleRefs);
% Handles to each toggle button's callback function.
handles.togCallbacks = ...
    containers.Map(handles.moduleNames, togCallbackRefs);
% Tracks the toggle state of each toggle button (must do this manually
% since we use push buttons instead; toggle buttons have uncontrollable
% color problems on Mac due to its default Java stylings).
handles.config.togValues = ...
    containers.Map(handles.moduleNames, ...
    num2cell(ones(1, size(handles.moduleNames, 2))));

% Set ReRef to be disabled by default.
togReRef_Callback(handles.togReRef, eventdata, handles);


% Update the value after the callback on reref.
handles.config.togValues('reref') = 0;

% Pre-Disabled Execute Screen GUI Items
handles.axes1.Visible = 'off';
handles.axes2.Visible = 'off';
handles.axes3.Visible = 'off';
handles.abort.Visible = 'off';
handles.saveLog.Visible = 'off';
handles.reconfig.Visible = 'off';
handles.commandWindow.Visible = 'off';
handles.myToolbar.Visible = 'off';

% Variable initialization
global abortTriggered;
abortTriggered = false;

% Initial files
handles.inputList.String = cell(0);
handles.files = cell(0);
handles.lastRadio = 0;

% Update handles structure
guidata(hObject, handles);

% check if there is an update
handles.commandWindowHandle = handles.inputList;
[~, msg] = CheckForUpdate(handles);

% Initial command window
handles.commandWindow.String = cellstr(msg);

% Update handles structure
guidata(hObject, handles);


% UIWAIT makes ReVAS wait for user response (see UIRESUME)
% uiwait(handles.revas);

% --- Outputs from this function are returned to the command line.
function varargout = Main_OutputFcn(hObject, eventdata, handles) %#ok<*INUSL>
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in parallelization.
function parallelization_Callback(hObject, eventdata, handles) %#ok<*INUSD,*DEFNU>
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

global abortTriggered;
abortTriggered = false;

% search for the file list content
filesSelected = false;
if ~isempty(handles.inputList.String)
    for ix = 1: length(handles.inputList.String)
        if ~isempty(strfind(handles.inputList.String{ix},'.avi')) || ...
           ~isempty(strfind(handles.inputList.String{ix},'.mat'))
            filesSelected = true;
            break;
        end
    end
end

% see if there are any files selected
if ~filesSelected
    parametersStructure.commandWindowHandle = handles.commandWindow;
    message = 'No files have been selected. Nothing to analyze.';
    RevasWarning(message, parametersStructure);
    % this is required since the ReVAS command window is not available at
    % this stage. but we have the file list box which we can use to put
    % this message on.
    handles.inputList.String = [message; handles.inputList.String];
    return;
end

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
handles.textReRef.Visible = 'off';
handles.textFilt.Visible = 'off';
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
handles.configReRef.Visible = 'off';
handles.configFilt.Visible = 'off';
handles.configSacDrift.Visible = 'off';
handles.parallelization.Visible = 'off';
handles.execute.Visible = 'off';
handles.reconfig.Visible = 'on';
handles.reconfig.Enable = 'off';
handles.myToolbar.Visible = 'on';

if handles.config.parMultiCore || handles.config.parGPU
    handles.axes1.Visible = 'off';
    handles.axes2.Visible = 'off';
    handles.axes3.Visible = 'off';
    handles.abort.Visible = 'off';
    handles.abort.Enable = 'off';
    handles.saveLog.Visible = 'off';
    handles.saveLog.Enable = 'off';
    handles.commandWindow.Visible = 'on';
else
    handles.axes1.Visible = 'on';
    handles.axes2.Visible = 'on';
    handles.axes3.Visible = 'on';
    handles.abort.Visible = 'on';
    handles.abort.Enable = 'on';
    handles.saveLog.Visible = 'on';
    handles.saveLog.Enable = 'off';
    handles.commandWindow.Visible = 'on';
end

handles.menuLoad.Enable = 'off';
handles.menuSave.Enable = 'off';

time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
handles.commandWindow.String = cellstr(['(' time ') Execution in Progress...']);
clc;
drawnow;

% Apply modules to all selected files
if logical(handles.config.parMultiCore)
    handles.commandWindow.String = ['Graphical verbosity is not available while parallelizing.'; ...
        handles.commandWindow.String];
    handles.commandWindow.String = ['Full output is being written to log.txt.'; ...
        handles.commandWindow.String];
    handles.commandWindow.String = ['Also see the Matlab command window for progress.'; ...
        handles.commandWindow.String];
    drawnow;
    if exist('log.txt', 'file') == 2
        delete 'log.txt';
    end
    fileID = fopen('log.txt', 'wt');
    diary log.txt;
    fprintf(['(' time ') Execution in Progress...\n']);
    % Use parallelization if requested
    % TODO deal with GPU (see |ExecuteModules.m|).
    parametersStructure.commandWindowHandle = handles.commandWindow;
    parfor i = 1:size(handles.files, 2)
        try
            % TODO perhaps use loop unrolling to suppress warning below
            ExecuteModules(handles.files{i}, handles); %#ok<PFBNS>
        catch ME
            message = [ME.message ' '];
            for j = 1:size(ME.stack, 1)
                message = [message ME.stack(j).name '(' int2str(ME.stack(j).line) ') < '];
            end
            message = [message(1:end-3) '.'];
            RevasError(handles.files{i},message,parametersStructure);
        end
    end
    fprintf(['(' time ') Process completed.\n']);
    diary off;
    fclose(fileID);
    % Parallel computing might print out of order.
    % Fix it by sorting the timestamps leading each line.
    !sort log.txt > temp.txt; mv temp.txt log.txt
else
    % Otherwise use a regular for loop
    for i = 1:size(handles.files, 2)
        if ~logical(abortTriggered)
            try
                ExecuteModules(handles.files{i}, handles);
            catch ME
                % Catch any errors that arise and display to output.
                parametersStructure.commandWindowHandle = handles.commandWindow;
                message = [ME.message ' '];
                for j = 1:size(ME.stack, 1)
                    message = [message ME.stack(j).name '(' int2str(ME.stack(j).line) ') < ']; %#ok<AGROW>
                end
                message = [message(1:end-3) '.'];
                RevasError(handles.files{i}, message, parametersStructure);
            end
        end
    end
end

time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
    
if logical(abortTriggered)
    handles.commandWindow.String = ['(' time ') Process aborted by user.'; ...
        handles.commandWindow.String];
    warndlg('Process aborted by user.', 'Process Aborted');
    handles.saveLog.Enable = 'on';
    handles.reconfig.Enable = 'on';
else   
    handles.commandWindow.String = ['(' time ') Process completed.'; ...
        handles.commandWindow.String];
    msgbox('Process completed.', 'Process Completed');
    handles.abort.Enable = 'off';
    handles.reconfig.Enable = 'on';
    handles.saveLog.Enable = 'on';
end

% Helper function called by radio button callback functions. Restores
% specified toggle and config button to state before they were disabled.
function handles = reEnable(index, eventdata, handles)
module = handles.moduleNames{index};
if strcmp(handles.togHandles(module).Enable, 'off')
    handles.config.togValues(module) = ...
        1 - handles.config.preDisabledTogValues(module);
    currHandle = handles.togHandles(module);
    currHandle.Enable = 'on';
    currHandle = handles.configHandles(module);
    currHandle.Enable = 'on';
    currCallback = handles.togCallbacks(module);
    currCallback(handles.togHandles(module), eventdata, handles);
    handles.config.togValues(module) = ...
        handles.config.preDisabledTogValues(module);
end

% Helper function called by radio button callback functions. Temporarily
% disables specified toggle and config button.
function handles = tempDisable(index, eventdata, handles)
module = handles.moduleNames{index};
if strcmp(handles.togHandles(module).Enable, 'on')
    handles.config.preDisabledTogValues(module) = ...
        handles.config.togValues(module);
    handles.config.togValues(module) = 1;
    currHandle = handles.togHandles(module);
    currHandle.Enable = 'off';
    currHandle = handles.configHandles(module);
    currHandle.Enable = 'off';
    currCallback = handles.togCallbacks(module);
    currCallback(handles.togHandles(module), eventdata, handles);
    handles.config.togValues(module) = 0;
end

% Helper function to apply callback on a toggle button.
function handles = toggle_Callback(module, handles)
handles.config.togValues(module) = 1 - handles.config.togValues(module);
currHandle = handles.togHandles(module);
if handles.config.togValues(module) == 1
%     currHandle.String = 'ENABLED';
    currHandle.BackgroundColor = handles.revasColors.activeButtonBackground;
    currHandle.ForegroundColor = handles.revasColors.activeButtonText;
else
%     currHandle.String = 'DISABLED';
    currHandle.BackgroundColor = handles.revasColors.passiveButtonBackground;
    currHandle.ForegroundColor = handles.revasColors.passiveButtonText;
end

% --- Executes on button press in radioRaw.
function radioRaw_Callback(hObject, eventdata, handles)
% hObject    handle to radioRaw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioRaw
% We update coarse after fine and strip to avoid warning dialogues.
for i = [1, 2, 3, 4, 6, 7, 5]
    handles = reEnable(i, eventdata, handles);
end

if handles.lastRadio ~= 1
    handles.lastRadio = 1;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
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
    ext = '.avi';
elseif get(handles.radioTrim, 'Value')
    suffix = '_dwt.avi';       
elseif get(handles.radioNoStim, 'Value')
    suffix = '_nostim.avi';  
elseif get(handles.radioGamma, 'Value')
    suffix = '_gamscaled.avi';   
elseif get(handles.radioBandFilt, 'Value')
    suffix = '_bandfilt.avi';
elseif get(handles.radioStrip, 'Value')
    suffix = '_hz_final.mat';
else
    suffix = '';
end

% for _final_ files, allow filtered files as well
if get(handles.radioStrip, 'Value')
    handles.files = uipickfiles('FilterSpec',['*' suffix(1:end-4) '*' suffix(end-3:end)]);
else
    handles.files = uipickfiles('FilterSpec',['*' suffix]);
end

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
            if ~strcmp(folderFiles(j).name, '.') && ...
                    ~strcmp(folderFiles(j).name, '..') && ...
                    (~isempty(strfind(folderFiles(j).name, suffix)) || ...
                    isdir(fullfile(folder, folderFiles(j).name)))
                handles.files = ...
                    [handles.files, {fullfile(folder, folderFiles(j).name)}];
            end
        end
    elseif isempty(strfind(handles.files{i}, suffix)) || ...
            (strcmp('.avi', suffix) && ...
                    (~isempty(findstr('_dwt', handles.files{i})) || ...
                    ~isempty(findstr('_nostim', handles.files{i})) || ...
                    ~isempty(findstr('_gamscaled', handles.files{i})) || ...
                    ~isempty(findstr('_bandfilt', handles.files{i})))) %#ok<*FSTR>
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
set(handles.inputList,'Value',1); 
for i = 1:size(handles.files, 2)
    [~,displayFileList{i},~] = fileparts(handles.files{i});
    if get(handles.radioStrip, 'Value')
        displayFileList{i} = [displayFileList{i} '.mat'];
    else
        displayFileList{i} = [displayFileList{i} '.avi'];
    end
end
handles.inputList.String = displayFileList;

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
handles = toggle_Callback('trim', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togStim.
function togStim_Callback(hObject, eventdata, handles)
handles = toggle_Callback('stim', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togGamma.
function togGamma_Callback(hObject, eventdata, handles)
handles = toggle_Callback('gamma', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togBandFilt.
function togBandFilt_Callback(hObject, eventdata, handles)
handles = toggle_Callback('bandfilt', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togCoarse.
function togCoarse_Callback(hObject, eventdata, handles)

% MNA 11/25/19 
% commented out the check for fineRef. CoarseRef is no longer
% a requirement for fineRef.

% % % First check to see if an illegal flip of coarse is taking place (i.e.
% % % disabling coarse while fine is still enabled, since a trigger on this
% % % callback means that there is an attempt to flip the state of coarse.)
% % if handles.config.togValues('coarse') == 1 && ...
% %     handles.config.togValues('fine') == 1 && ...
% %     handles.radioStrip.Value ~= 1
% %     errordlg(...
% %         'Make Coarse Reference Frame must be enabled if Make Fine Reference Frame is enabled.', 'Invalid Selection');
% %     handles.config.togValues('coarse') = 1;
% % else
% %     handles = toggle_Callback('coarse', handles);
% % end

handles = toggle_Callback('coarse', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togFine.
function togFine_Callback(hObject, eventdata, handles)

% MNA 11/25/19 
% commented out the check for CoarseRef. It's no longer a requirement.

% % % First check to see if an illegal flip of fine is taking place (i.e.
% % % enabling fine while coarse is still disabled, since a trigger on this
% % % callback means that there is an attempt to flip the state of fine.)
% % if handles.config.togValues('fine') == 0 && ...
% %     handles.config.togValues('coarse') == 0 && ...
% %     handles.lastRadio ~= 6
% %         warndlg('Make Coarse Reference Frame has been enabled since it must be if Make Fine Reference Frame is enabled.', 'Input Warning');
% %         handles.config.togValues('coarse') = 0;
% %         togCoarse_Callback(handles.togCoarse, eventdata, handles);
% %         handles.config.togValues('coarse') = 1;
% % end

handles = toggle_Callback('fine', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togStrip.
function togStrip_Callback(hObject, eventdata, handles)
handles = toggle_Callback('strip', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togReRef.
function togReRef_Callback(hObject, eventdata, handles)
handles = toggle_Callback('reref', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togFilt.
function togFilt_Callback(hObject, eventdata, handles)
handles = toggle_Callback('filt', handles);

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in togSacDrift.
function togSacDrift_Callback(hObject, eventdata, handles)
handles = toggle_Callback('sacdrift', handles);

% Update handles structure
guidata(hObject, handles);

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

% --- Executes on button press in configCoarse.
function configCoarse_Callback(hObject, eventdata, handles)
% hObject    handle to configCoarse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
CoarseParameters;

% --- Executes on button press in configFine.
function configFine_Callback(hObject, eventdata, handles)
% hObject    handle to configFine (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FineParameters;

% --- Executes on button press in configStrip.
function configStrip_Callback(hObject, eventdata, handles)
% hObject    handle to configStrip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
StripParameters;

% --- Executes on button press in configReRef.
function configReRef_Callback(hObject, eventdata, handles)
% hObject    handle to configFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ReRefParameters;

% --- Executes on button press in configFilt.
function configFilt_Callback(hObject, eventdata, handles)
% hObject    handle to configFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FilteringParameters;

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

% We update coarse after fine and strip to avoid warning dialogues.
for i = 1
    handles = tempDisable(i, eventdata, handles);
end
for i = [2, 3, 4, 6, 7, 5]
    handles = reEnable(i, eventdata, handles);
end

if handles.lastRadio ~= 2
    handles.lastRadio = 2;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioNoStim.
function radioNoStim_Callback(hObject, eventdata, handles)
% hObject    handle to radioNoStim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioNoStim

% We update coarse after fine and strip to avoid warning dialogues.
for i = [1, 2]
    handles = tempDisable(i, eventdata, handles);
end
for i = [3, 4, 6, 7, 5]
    handles = reEnable(i, eventdata, handles);
end

if handles.lastRadio ~= 3
    handles.lastRadio = 3;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioGamma.
function radioGamma_Callback(hObject, eventdata, handles)
% hObject    handle to radioGamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioGamma

% We update coarse after fine and strip to avoid warning dialogues.
for i = [1, 2, 3]
    handles = tempDisable(i, eventdata, handles);
end
for i = [4, 6, 7, 5]
    handles = reEnable(i, eventdata, handles);
end

if handles.lastRadio ~= 4
    handles.lastRadio = 4;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioBandFilt.
function radioBandFilt_Callback(hObject, eventdata, handles)
% hObject    handle to radioBandFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioBandFilt

% We update coarse after fine and strip to avoid warning dialogues.
for i = [1, 2, 3, 4]
    handles = tempDisable(i, eventdata, handles);
end
for i = [6, 7, 5]
    handles = reEnable(i, eventdata, handles);
end

if handles.lastRadio ~= 5
    handles.lastRadio = 5;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in radioStrip.
function radioStrip_Callback(hObject, eventdata, handles)
% hObject    handle to radioStrip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioStrip

for i = [1, 2, 3, 4, 6, 7, 5]
    handles = tempDisable(i, eventdata, handles);
end

if handles.lastRadio ~= 6
    handles.lastRadio = 6;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in abort.
function abort_Callback(hObject, eventdata, handles)
% hObject    handle to abort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global abortTriggered;
abortTriggered = true;

hObject.Enable = 'off';

% --------------------------------------------------------------------
function menuAbout_Callback(hObject, eventdata, handles)
% hObject    handle to menuAbout (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% % [isUpdated, newestVersion] = IsUpToDate();

% read the current version
fileID = fopen('README.md');
tline = fgetl(fileID);


message = ['Retinal Video Analysis Suite (' tline ')' newline newline...
    'Copyright (c) 2018.' newline ...
    'Sight Enhancement Laboratory at Berkeley.' newline ...
    'School of Optometry.' newline ...
    'University of California, Berkeley, USA.' newline newline ...
    'Mehmet N. Agaoglu, PhD.' newline ...
    'mnagaoglu@gmail.com' newline ...
    'Matthew T. Sit.' newline ...
    'msit@berkeley.edu' newline ...
    'Derek Wan.' newline ...
    'derek.wan11@berkeley.edu' newline newline ...
    'Susana T. L. Chung, OD, PhD.' newline ...
    's.chung@berkeley.edu'];

% if ~isUpdated
%     message{end+1} = '';
%     message{end+1} = strcat('The newest version of ReVAS (', ...
%         newestVersion, ...
%         ') is now available. Please visit GitHub to update your software.');
% end

figure('Name', 'About ReVAS','units','normalized',...
    'outerposition',[0.2738    0.4162    0.1929    0.4029],...
    'MenuBar','none','ToolBar','none');
ax = axes('position',[0 0 1 1]);
ax.XAxis.Color = handles.revasColors.background;
ax.YAxis.Color = handles.revasColors.background;
text(.08,.5,message,'Color',handles.revasColors.text,...
    'fontsize',handles.uiFontSize);




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
toggleButtonStates = containers.Map;

% Suppress warning if variables not found in loaded file
warning('off','MATLAB:load:variableNotFound');

load(fullfile(pathName, fileName), 'configurationsStruct', 'toggleButtonStates');

% Copying over data (values not provided will remain unchanged)
configurationsStructFieldNames = fieldnames(configurationsStruct);
for i = 1:length(configurationsStructFieldNames)
    handles.config.(configurationsStructFieldNames{i}) = ...
        configurationsStruct.(configurationsStructFieldNames{i});
end

% Switch to raw radio button first to avoid illegal states.
radioRaw_Callback(handles.radioRaw, eventdata, handles);
handles.radioRaw.Value = 1;

try
    % To avoid error dialogues, must be careful with Coarse and Fine toggle
    % states when restoring previous saved values.
    
    % Copy over inverted values to prepare for callbacks.
    for i = 1:size(handles.moduleNames, 2)
        module = handles.moduleNames{i};
        handles.config.togValues(module) = 1 - toggleButtonStates(module);
    end
    
    % Start with the first 4 callbacks.
    for i = 1:4
        module = handles.moduleNames{i};
        currCallback = handles.togCallbacks(module);
        currCallback(handles.togHandles(module), eventdata, handles);
    end
    
    % Set fine and coarse to saved config carefully.
    handles.config.togValues('coarse') = toggleButtonStates('coarse');
    handles.config.togValues('fine') = toggleButtonStates('fine');
    handles.config.togValues('fine') = 1 - toggleButtonStates('fine');
    togFine_Callback(handles.togHandles('fine'), eventdata, handles);
    handles.config.togValues('fine') = toggleButtonStates('fine');
    handles.config.togValues('coarse') = 1 - toggleButtonStates('coarse');
    togCoarse_Callback(handles.togHandles('coarse'), eventdata, handles);
    handles.config.togValues('coarse') = toggleButtonStates('coarse');
    
    % Finish the remaining callbacks.
    for i = 7:size(handles.moduleNames, 2)
        module = handles.moduleNames{i};
        currCallback = handles.togCallbacks(module);
        currCallback(handles.togHandles(module), eventdata, handles);
    end
    
    % Save back correct values
    for i = 1:size(handles.moduleNames, 2)
        module = handles.moduleNames{i};
        handles.config.togValues(module) = toggleButtonStates(module);
    end
catch
    errordlg('Load configurations failed because the file is corrupted or incompatible with the current release.', ...
        'Corrupt Configurations File');
    return;
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
toggleButtonStates = containers.Map(...
    keys(handles.config.togValues), ...
    values(handles.config.togValues));

for i = 1:7
    module = handles.moduleNames{i};
    currHandle = handles.togHandles(module);
    if strcmp(currHandle.Enable, 'off')
        toggleButtonStates(module) = handles.config.preDisabledTogValues(module);
    end
end

save(fullfile(pathName, fileName), 'configurationsStruct', 'toggleButtonStates');

% --------------------------------------------------------------------
function menuExit_Callback(hObject, eventdata, handles)
% hObject    handle to menuExit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


close;

% --- Executes during object creation, after setting all properties.
function commandWindow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to commandWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in reconfig.
function reconfig_Callback(hObject, eventdata, handles)
% hObject    handle to reconfig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Abort first
global abortTriggered;
if ~logical(abortTriggered)
    abort_Callback(handles.abort, eventdata, handles);
end

% Clear figures, changing from surf to scalar before clearing.
axes(handles.axes1);
cla reset;
drawnow;
axes(handles.axes2);
cla reset;
drawnow;
axes(handles.axes3);
cla reset;
drawnow;

handles.axes1.XColor = handles.revasColors.background;
handles.axes1.YColor = handles.revasColors.background;
handles.axes2.XColor = handles.revasColors.background;
handles.axes2.YColor = handles.revasColors.background;
handles.axes3.XColor = handles.revasColors.background;
handles.axes3.YColor = handles.revasColors.background;
set(handles.axes1,'fontunits','points','fontsize',handles.uiFontSize);
set(handles.axes2,'fontunits','points','fontsize',handles.uiFontSize);
set(handles.axes3,'fontunits','points','fontsize',handles.uiFontSize);

drawnow;

% Update visible and invisible gui components
handles.inputVideoBox.Visible = 'on';
handles.selectFiles.Visible = 'on';
handles.inputList.Visible = 'on';
handles.modulesBox.Visible = 'on';
handles.textTrim.Visible = 'on';
handles.textStim.Visible = 'on';
handles.textGamma.Visible = 'on';
handles.textBandFilt.Visible = 'on';
handles.textCoarse.Visible = 'on';
handles.textFine.Visible = 'on';
handles.textStrip.Visible = 'on';
handles.textReRef.Visible = 'on';
handles.textFilt.Visible = 'on';
handles.textSacDrift.Visible = 'on';
handles.togTrim.Visible = 'on';
handles.togStim.Visible = 'on';
handles.togGamma.Visible = 'on';
handles.togBandFilt.Visible = 'on';
handles.togCoarse.Visible = 'on';
handles.togFine.Visible = 'on';
handles.togStrip.Visible = 'on';
handles.togFilt.Visible = 'on';
handles.togReRef.Visible = 'on';
handles.togSacDrift.Visible = 'on';
handles.configTrim.Visible = 'on';
handles.configStim.Visible = 'on';
handles.configGamma.Visible = 'on';
handles.configBandFilt.Visible = 'on';
handles.configCoarse.Visible = 'on';
handles.configFine.Visible = 'on';
handles.configStrip.Visible = 'on';
handles.configReRef.Visible = 'on';
handles.configFilt.Visible = 'on';
handles.configSacDrift.Visible = 'on';
handles.parallelization.Visible = 'on';
handles.execute.Visible = 'on';

handles.axes1.Visible = 'off';
handles.axes2.Visible = 'off';
handles.axes3.Visible = 'off';
handles.abort.Visible = 'off';
handles.saveLog.Visible = 'off';
handles.reconfig.Visible = 'off';
handles.commandWindow.Visible = 'off';

handles.menuLoad.Enable = 'on';
handles.menuSave.Enable = 'on';

handles.myToolbar.Visible = 'off';

drawnow;

% --- Executes on button press in saveLog.
function saveLog_Callback(hObject, eventdata, handles)
% hObject    handle to saveLog (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[fileName,pathName, ~] = uiputfile('*.txt','Save Log','log.txt');
if fileName == 0
    % User canceled.
    return;
end

fileID = fopen(fullfile(pathName, fileName), 'wt');
for i = size(handles.commandWindow.String):-1:1
    fprintf(fileID, '%s\n', handles.commandWindow.String{i});
end
fclose(fileID);

msgbox('Log saved.', 'Log Saved');

% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isvalid(hObject)
    axes(hObject);
end

% --- Executes on mouse press over axes background.
function axes2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isvalid(hObject)
    axes(hObject);
end

% --- Executes on mouse press over axes background.
function axes3_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isvalid(hObject)
    axes(hObject);
end



function revasCloseRequest(hObject,src,callbackdata)
% Close request function 
% to display a question dialog box 
selection = questdlg('Do you want to exit ReVAS?',...
              'Close Request',...
              'Yes','No','Yes'); 
          
switch selection 
    case 'Yes'
        objs = hObject.Parent.Children;
        delete(objs);
    case 'No'
        return 
end
