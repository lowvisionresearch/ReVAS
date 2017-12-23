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

% Last Modified by GUIDE v2.5 27-Nov-2017 00:25:35

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

% Choose default command line output for ReVAS
handles.output = hObject;

% Add util to path
addpath(genpath('..'));

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
% ReVAS Background
handles.revas.Color = handles.colors{1,2};
handles.inputList.BackgroundColor = handles.colors{1,2};
handles.axes1.XColor = handles.colors{1,2};
handles.axes1.YColor = handles.colors{1,2};
handles.axes2.XColor = handles.colors{1,2};
handles.axes2.YColor = handles.colors{1,2};
handles.axes3.XColor = handles.colors{1,2};
handles.axes3.YColor = handles.colors{1,2};
handles.commandWindow.BackgroundColor = handles.colors{1,2};

% Box backgrounds
handles.inputVideoBox.BackgroundColor = handles.colors{1,3};
handles.radioRaw.BackgroundColor = handles.colors{1,3};
handles.radioTrim.BackgroundColor = handles.colors{1,3};
handles.radioNoStim.BackgroundColor = handles.colors{1,3};
handles.radioGamma.BackgroundColor = handles.colors{1,3};
handles.radioBandFilt.BackgroundColor = handles.colors{1,3};
handles.radioStrip.BackgroundColor = handles.colors{1,3};
handles.modulesBox.BackgroundColor = handles.colors{1,3};
handles.textTrim.BackgroundColor = handles.colors{1,3};
handles.textStim.BackgroundColor = handles.colors{1,3};
handles.textGamma.BackgroundColor = handles.colors{1,3};
handles.textBandFilt.BackgroundColor = handles.colors{1,3};
handles.textCoarse.BackgroundColor = handles.colors{1,3};
handles.textFine.BackgroundColor = handles.colors{1,3};
handles.textStrip.BackgroundColor = handles.colors{1,3};
handles.textReRef.BackgroundColor = handles.colors{1,3};
handles.textFilt.BackgroundColor = handles.colors{1,3};
handles.textSacDrift.BackgroundColor = handles.colors{1,3};
% Box text
handles.inputList.ForegroundColor = handles.colors{1,5};
handles.inputVideoBox.ForegroundColor = handles.colors{1,5};
handles.radioRaw.ForegroundColor = handles.colors{1,5};
handles.radioTrim.ForegroundColor = handles.colors{1,5};
handles.radioNoStim.ForegroundColor = handles.colors{1,5};
handles.radioGamma.ForegroundColor = handles.colors{1,5};
handles.radioBandFilt.ForegroundColor = handles.colors{1,5};
handles.radioStrip.ForegroundColor = handles.colors{1,5};
handles.modulesBox.ForegroundColor = handles.colors{1,5};
handles.textTrim.ForegroundColor = handles.colors{1,5};
handles.textStim.ForegroundColor = handles.colors{1,5};
handles.textGamma.ForegroundColor = handles.colors{1,5};
handles.textBandFilt.ForegroundColor = handles.colors{1,5};
handles.textCoarse.ForegroundColor = handles.colors{1,5};
handles.textFine.ForegroundColor = handles.colors{1,5};
handles.textStrip.ForegroundColor = handles.colors{1,5};
handles.textReRef.ForegroundColor = handles.colors{1,5};
handles.textFilt.ForegroundColor = handles.colors{1,5};
handles.textSacDrift.ForegroundColor = handles.colors{1,5};
handles.commandWindow.ForegroundColor = handles.colors{1,5};
% Select/Enable buttons backgrounds
handles.selectFiles.BackgroundColor = handles.colors{1,4};
handles.togTrim.BackgroundColor = handles.colors{1,4};
handles.togStim.BackgroundColor = handles.colors{1,4};
handles.togGamma.BackgroundColor = handles.colors{1,4};
handles.togBandFilt.BackgroundColor = handles.colors{1,4};
handles.togCoarse.BackgroundColor = handles.colors{1,4};
handles.togFine.BackgroundColor = handles.colors{1,4};
handles.togStrip.BackgroundColor = handles.colors{1,4};
handles.togReRef.BackgroundColor = handles.colors{1,4};
handles.togFilt.BackgroundColor = handles.colors{1,4};
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
handles.togReRef.ForegroundColor = handles.colors{1,2};
handles.togFilt.ForegroundColor = handles.colors{1,2};
handles.togSacDrift.ForegroundColor = handles.colors{1,2};
% Configure buttons backgrounds
handles.configTrim.BackgroundColor = handles.colors{4,4};
handles.configStim.BackgroundColor = handles.colors{4,4};
handles.configGamma.BackgroundColor = handles.colors{4,4};
handles.configBandFilt.BackgroundColor = handles.colors{4,4};
handles.configCoarse.BackgroundColor = handles.colors{4,4};
handles.configFine.BackgroundColor = handles.colors{4,4};
handles.configStrip.BackgroundColor = handles.colors{4,4};
handles.configReRef.BackgroundColor = handles.colors{4,4};
handles.configFilt.BackgroundColor = handles.colors{4,4};
handles.configSacDrift.BackgroundColor = handles.colors{4,4};
% Configure button text
handles.configTrim.ForegroundColor = handles.colors{4,2};
handles.configStim.ForegroundColor = handles.colors{4,2};
handles.configGamma.ForegroundColor = handles.colors{4,2};
handles.configBandFilt.ForegroundColor = handles.colors{4,2};
handles.configCoarse.ForegroundColor = handles.colors{4,2};
handles.configFine.ForegroundColor = handles.colors{4,2};
handles.configStrip.ForegroundColor = handles.colors{4,2};
handles.configReRef.ForegroundColor = handles.colors{4,2};
handles.configFilt.ForegroundColor = handles.colors{4,2};
handles.configSacDrift.ForegroundColor = handles.colors{4,2};
% Parallelization button background
handles.parallelization.BackgroundColor = handles.colors{4,4};
% Parallelization button text
handles.parallelization.ForegroundColor = handles.colors{4,2};
% Execute button background
handles.execute.BackgroundColor = handles.colors{3,4};
% Execute button text
handles.execute.ForegroundColor = handles.colors{3,2};
% Save Log button background
handles.saveLog.BackgroundColor = handles.colors{4,4};
% Save Log button text
handles.saveLog.ForegroundColor = handles.colors{4,2};
% Re-Config button background
handles.reconfig.BackgroundColor = handles.colors{4,4};
% Re-Config button text
handles.reconfig.ForegroundColor = handles.colors{4,2};

% DEFAULT PARAMETERS
% Trim
handles.config.trimLeft = 0;
handles.config.trimRight = 24;
handles.config.trimTop = 24;
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
% BandFilt
handles.config.bandFiltSmoothing = 1;
handles.config.bandFiltFreqCut = 3.0;
handles.config.bandFiltOverwrite = true;
% Coarse
handles.config.coarseRefFrameNum = 15;
handles.config.coarseScalingFactor = 1.0;
handles.config.coarseOverwrite = true;
handles.config.coarseVerbosity = true;
% Fine
handles.config.fineOverwrite = true;
handles.config.fineVerbosity = true;
handles.config.fineNumIterations = 1;
handles.config.fineStripHeight = 15;
handles.config.fineStripWidth = 488;
handles.config.fineSamplingRate = 540;
handles.config.fineMaxPeakRatio = 0.8;
handles.config.fineMinPeakThreshold = 0.2;
handles.config.fineAdaptiveSearch = false;
handles.config.fineScalingFactor = 8;
handles.config.fineSearchWindowHeight = 79;
handles.config.fineSubpixelInterp = false;
handles.config.fineNeighborhoodSize = 7;
handles.config.fineSubpixelDepth = 2;
% Strip
handles.config.stripCreateStabilizedVideo = false;
handles.config.stripOverwrite = true;
handles.config.stripVerbosity = true;
handles.config.stripStripHeight = 15;
handles.config.stripStripWidth = 488;
handles.config.stripSamplingRate = 540;
handles.config.stripEnableGaussFilt = false;
handles.config.stripDisableGaussFilt = true;
handles.config.stripGaussSD = 10;
handles.config.stripSDWindow = 25;
handles.config.stripMaxPeakRatio = 0.8;
handles.config.stripMinPeakThreshold = 0;
handles.config.stripAdaptiveSearch = false;
handles.config.stripScalingFactor = 8;
handles.config.stripSearchWindowHeight = 79;
handles.config.stripSubpixelInterp = true;
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

% Pre-Disabled Toggle Values
handles.config.preDisabledTogTrimValue = 1;
handles.config.preDisabledTogStimValue = 1;
handles.config.preDisabledTogGammaValue = 1;
handles.config.preDisabledTogBandFiltValue = 1;
handles.config.preDisabledTogCoarseValue = 1;
handles.config.preDisabledTogFineValue = 1;
handles.config.preDisabledTogStripValue = 1;
handles.togReRef.Value = 0;
togReRef_Callback(handles.togReRef, eventdata, handles);

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

% Initial command window
handles.commandWindow.String = cellstr('');

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ReVAS wait for user response (see UIRESUME)
% uiwait(handles.revas);

% --- Outputs from this function are returned to the command line.
function varargout = Main_OutputFcn(hObject, eventdata, handles)
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

global abortTriggered;
abortTriggered = false;

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

dateAndTime = datestr(datetime('now'));
time = dateAndTime(13:20);
handles.commandWindow.String = cellstr(['(' time ') Execution in Progress...']);
clc;
drawnow;

% Apply modules to all selected files
if logical(handles.config.parMultiCore)
    handles.commandWindow.String = ['Verbosity is not available while parallelizing.'; ...
        handles.commandWindow.String];
    handles.commandWindow.String = ['Full output is being written to log.txt.'; ...
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
    parfor i = 1:size(handles.files, 2)
        try
            % TODO perhaps use loop unrolling to suppress warning below
            ExecuteModules(handles.files{i}, handles);
        catch ME
            message = [ME.message ' '];
            for j = 1:size(ME.stack, 1)
                message = [message ME.stack(j).name '(' int2str(ME.stack(j).line) ') < '];
            end
            message = [message(1:end-3) '.'];
            warning(['(Error while processing ' handles.files{i} '. Proceeding to next video.) ' ...
                message]);
        end
    end
    fprintf(['(' time ') Process completed.\n']);
    diary off;
    fclose(fileID);
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
                    message = [message ME.stack(j).name '(' int2str(ME.stack(j).line) ') < '];
                end
                message = [message(1:end-3) '.'];
                RevasError(handles.files{i}, message, parametersStructure);
            end
        end
    end
end

dateAndTime = datestr(datetime('now'));
time = dateAndTime(13:20);
    
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

if strcmp(handles.togFine.Enable, 'off')
    handles.togFine.Value = handles.config.preDisabledTogFineValue;
    handles.togFine.Enable = 'on';
    handles.configFine.Enable = 'on';
    togFine_Callback(handles.togFine, eventdata, handles);
end

if strcmp(handles.togStrip.Enable, 'off')
    handles.togStrip.Value = handles.config.preDisabledTogStripValue;
    handles.togStrip.Enable = 'on';
    handles.configStrip.Enable = 'on';
    togStrip_Callback(handles.togStrip, eventdata, handles);
end

if strcmp(handles.togCoarse.Enable, 'off')
    handles.togCoarse.Value = handles.config.preDisabledTogCoarseValue;
    handles.togCoarse.Enable = 'on';
    handles.configCoarse.Enable = 'on';
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
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

handles.files = uipickfiles();

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
            if ~isempty(findstr(folderFiles(j).name, suffix)) || ...
                    isdir(fullfile(folder, folderFiles(j).name))
                handles.files = ...
                    [handles.files, {fullfile(folder, folderFiles(j).name)}];
            end
        end
    elseif isempty(findstr(handles.files{i}, suffix)) || ...
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
    if get(handles.radioStrip, 'Value')
        displayFileList{i} = [displayFileList{i} '.mat'];
    else
        displayFileList{i} = [displayFileList{i} '.avi'];
    end
end
handles.inputList.String = displayFileList;

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

if strcmp(handles.togFine.Enable, 'off')
    handles.togFine.Value = handles.config.preDisabledTogFineValue;
    handles.togFine.Enable = 'on';
    handles.configFine.Enable = 'on';
    togFine_Callback(handles.togFine, eventdata, handles);
end

if strcmp(handles.togStrip.Enable, 'off')
    handles.togStrip.Value = handles.config.preDisabledTogStripValue;
    handles.togStrip.Enable = 'on';
    handles.configStrip.Enable = 'on';
    togStrip_Callback(handles.togStrip, eventdata, handles);
end

if strcmp(handles.togCoarse.Enable, 'off')
    handles.togCoarse.Value = handles.config.preDisabledTogCoarseValue;
    handles.togCoarse.Enable = 'on';
    handles.configCoarse.Enable = 'on';
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
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

if strcmp(handles.togCoarse.Enable, 'on')
    handles.config.preDisabledTogCoarseValue = handles.togCoarse.Value;
end
handles.togCoarse.Value = 0;
handles.togCoarse.Enable = 'off';
handles.configCoarse.Enable = 'off';
togCoarse_Callback(handles.togCoarse, eventdata, handles);

if strcmp(handles.togFine.Enable, 'on')
    handles.config.preDisabledTogFineValue = handles.togFine.Value;
end
handles.togFine.Value = 0;
handles.togFine.Enable = 'off';
handles.configFine.Enable = 'off';
togFine_Callback(handles.togFine, eventdata, handles);

if strcmp(handles.togStrip.Enable, 'on')
    handles.config.preDisabledTogStripValue = handles.togStrip.Value;
end
handles.togStrip.Value = 0;
handles.togStrip.Enable = 'off';
handles.configStrip.Enable = 'off';
togStrip_Callback(handles.togStrip, eventdata, handles);

if handles.lastRadio ~= 6
    handles.lastRadio = 6;
    handles.inputList.String = cell(0);
    handles.files = cell(0);
end

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
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togStrip.
function togStrip_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togStim.
function togStim_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togGamma.
function togGamma_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togCoarse.
function togCoarse_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    if handles.togFine.Value == 1 && handles.radioStrip.Value ~= 1
        errordlg(...
            'Make Coarse Reference Frame must be enabled if Make Fine Reference Frame is enabled.', 'Invalid Selection');
        hObject.Value = 1;
        return;
    end
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togReRef.
function togReRef_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togFilt.
function togFilt_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togSacDrift.
function togSacDrift_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togFine.
function togFine_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    if handles.togCoarse.Value == 0 && handles.lastRadio ~= 6
        warndlg('Make Coarse Reference Frame has been enabled since it must be if Make Fine Reference Frame is enabled.', 'Input Warning');
        handles.togCoarse.Value = 1;
        togCoarse_Callback(handles.togCoarse, eventdata, handles);
    end
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
end

% --- Executes on button press in togBandFilt.
function togBandFilt_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = handles.colors{1,4};
    hObject.ForegroundColor = handles.colors{1,2};
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = handles.colors{1,1};
    hObject.ForegroundColor = handles.colors{1,3};
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
FilteringParameters;

% --- Executes on button press in configReRef.
function configReRef_Callback(hObject, eventdata, handles)
% hObject    handle to configFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ReRefParameters;

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

if strcmp(handles.togFine.Enable, 'off')
    handles.togFine.Value = handles.config.preDisabledTogFineValue;
    handles.togFine.Enable = 'on';
    handles.configFine.Enable = 'on';
    togFine_Callback(handles.togFine, eventdata, handles);
end

if strcmp(handles.togStrip.Enable, 'off')
    handles.togStrip.Value = handles.config.preDisabledTogStripValue;
    handles.togStrip.Enable = 'on';
    handles.configStrip.Enable = 'on';
    togStrip_Callback(handles.togStrip, eventdata, handles);
end

if strcmp(handles.togCoarse.Enable, 'off')
    handles.togCoarse.Value = handles.config.preDisabledTogCoarseValue;
    handles.togCoarse.Enable = 'on';
    handles.configCoarse.Enable = 'on';
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
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

if strcmp(handles.togFine.Enable, 'off')
    handles.togFine.Value = handles.config.preDisabledTogFineValue;
    handles.togFine.Enable = 'on';
    handles.configFine.Enable = 'on';
    togFine_Callback(handles.togFine, eventdata, handles);
end

if strcmp(handles.togStrip.Enable, 'off')
    handles.togStrip.Value = handles.config.preDisabledTogStripValue;
    handles.togStrip.Enable = 'on';
    handles.configStrip.Enable = 'on';
    togStrip_Callback(handles.togStrip, eventdata, handles);
end

if strcmp(handles.togCoarse.Enable, 'off')
    handles.togCoarse.Value = handles.config.preDisabledTogCoarseValue;
    handles.togCoarse.Enable = 'on';
    handles.configCoarse.Enable = 'on';
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
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

if strcmp(handles.togCoarse.Enable, 'off')
    handles.togCoarse.Value = handles.config.preDisabledTogCoarseValue;
    handles.togCoarse.Enable = 'on';
    handles.configCoarse.Enable = 'on';
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
end

if strcmp(handles.togFine.Enable, 'off')
    handles.togFine.Value = handles.config.preDisabledTogFineValue;
    handles.togFine.Enable = 'on';
    handles.configFine.Enable = 'on';
    togFine_Callback(handles.togFine, eventdata, handles);
end

if strcmp(handles.togStrip.Enable, 'off')
    handles.togStrip.Value = handles.config.preDisabledTogStripValue;
    handles.togStrip.Enable = 'on';
    handles.configStrip.Enable = 'on';
    togStrip_Callback(handles.togStrip, eventdata, handles);
end

if handles.lastRadio ~= 4
    handles.lastRadio = 4;
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
msgbox({'Retinal Video Analysis Suite (ReVAS)'; ...
    'Copyright (c) August 2017.'; ...
    'Sight Enhancement Laboratory at Berkeley.'; ...
    'School of Optometry.'; ...
    'University of California, Berkeley, USA.'; ...
    ''; ...
    'Mehmet N. Agaoglu, PhD.'; ...
    'mnagaoglu@gmail.com'; ...
    ''; ...
    'Matthew T. Sit.'; ...
    'msit@berkeley.edu'; ...
    ''; ...
    'Derek Wan.'; ...
    'derek.wan11@berkeley.edu'; ...
    ''; ...
    'Susana T. L. Chung, OD, PhD.'; ...
    's.chung@berkeley.edu'}, ...
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
try
    handles.togTrim.Value = toggleButtonStates('togTrim');
    handles.togStim.Value = toggleButtonStates('togStim');
    handles.togGamma.Value = toggleButtonStates('togGamma');
    handles.togBandFilt.Value = toggleButtonStates('togBandFilt');
    handles.togCoarse.Value = toggleButtonStates('togCoarse');
    handles.togFine.Value = toggleButtonStates('togFine');
    handles.togStrip.Value = toggleButtonStates('togStrip');
    handles.togReRef.Value = toggleButtonStates('togReRef');
    handles.togFilt.Value = toggleButtonStates('togFilt');
    handles.togSacDrift.Value = toggleButtonStates('togSacDrift');
    
    togTrim_Callback(handles.togTrim, eventdata, handles);
    togStim_Callback(handles.togStim, eventdata, handles);
    togGamma_Callback(handles.togGamma, eventdata, handles);
    togBandFilt_Callback(handles.togBandFilt, eventdata, handles);
    togCoarse_Callback(handles.togCoarse, eventdata, handles);
    togFine_Callback(handles.togFine, eventdata, handles);
    togStrip_Callback(handles.togStrip, eventdata, handles);
    togReRef_Callback(handles.togReRef, eventdata, handles);
    togFilt_Callback(handles.togFilt, eventdata, handles);
    togSacDrift_Callback(handles.togSacDrift, eventdata, handles);
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
toggleButtonStates = containers.Map;
toggleButtonStates('togTrim') = handles.togTrim.Value;
toggleButtonStates('togStim') = handles.togStim.Value;
toggleButtonStates('togGamma') = handles.togGamma.Value;
toggleButtonStates('togBandFilt') = handles.togBandFilt.Value;
toggleButtonStates('togCoarse') = handles.togCoarse.Value;
toggleButtonStates('togFine') = handles.togFine.Value;
toggleButtonStates('togStrip') = handles.togStrip.Value;
toggleButtonStates('togReRef') = handles.togReRef.Value;
toggleButtonStates('togFilt') = handles.togFilt.Value;
toggleButtonStates('togSacDrift') = handles.togSacDrift.Value;
if strcmp(handles.togTrim.Enable, 'off')
    toggleButtonStates('togTrim') = handles.config.preDisabledTogTrimValue;
end
if strcmp(handles.togStim.Enable, 'off')
    toggleButtonStates('togTrim') = handles.config.preDisabledTogStimValue;
end
if strcmp(handles.togGamma.Enable, 'off')
   toggleButtonStates('togGamma') = handles.config.preDisabledTogGammaValue;
end
if strcmp(handles.togBandFilt.Enable, 'off')
   toggleButtonStates('togBandFilt') = handles.config.preDisabledTogBandFiltValue;
end
if strcmp(handles.togCoarse.Enable, 'off')
   toggleButtonStates('togCoarse') = handles.config.preDisabledTogCoarseValue;
end
if strcmp(handles.togFine.Enable, 'off')
   toggleButtonStates('togFine') = handles.config.preDisabledTogFineValue;
end
if strcmp(handles.togStrip.Enable, 'off')
   toggleButtonStates('togStrip') = handles.config.preDisabledTogStripValue; %#ok<NASGU>
end
save(fullfile(pathName, fileName), 'configurationsStruct', 'toggleButtonStates');

% --------------------------------------------------------------------
function menuExit_Callback(hObject, eventdata, handles)
% hObject    handle to menuExit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;


function commandWindow_Callback(hObject, eventdata, handles)
% hObject    handle to commandWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of commandWindow as text
%        str2double(get(hObject,'String')) returns contents of commandWindow as a double


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

handles.axes1.XColor = handles.colors{1,2};
handles.axes1.YColor = handles.colors{1,2};
handles.axes2.XColor = handles.colors{1,2};
handles.axes2.YColor = handles.colors{1,2};
handles.axes3.XColor = handles.colors{1,2};
handles.axes3.YColor = handles.colors{1,2};

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
if isValid(hObject)
    figure(hObject);
end

% --- Executes on mouse press over axes background.
function axes2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isValid(hObject)
    figure(hObject);
end

% --- Executes on mouse press over axes background.
function axes3_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isValid(hObject)
    figure(hObject);
end

% --- Executes during object creation, after setting all properties.
function revas_CreateFcn(hObject, eventdata, handles)
% hObject    handle to revas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
