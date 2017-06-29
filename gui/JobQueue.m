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

% Last Modified by GUIDE v2.5 27-Jun-2017 21:06:05

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

% Disabling buttons for their initial states
set(handles.parallelization, 'enable', 'off'); % Not implemented yet

% DEFAULT PARAMETERS
% Trim
handles.trimBorderTrimAmount = 24;
handles.trimOverwrite = true;
% Stim
handles.stimVerbosity = true;
handles.stimOverwrite = true;
% Gamma
handles.gammaExponent = 0.6;
handles.gammaOverwrite = true;
% BandFilt
handles.bandFiltSmoothing = 1;
handles.bandFiltFreqCut = 3.0;
handles.bandFiltOverwrite = true;
% Coarse
handles.coarseRefFrameNum = 15;
handles.coarseScalingFactor = 0.5;
handles.coarseOverwrite = true;
handles.coarseVerbosity = true;
% Fine
handles.fineOverwrite = true;
handles.fineVerbosity = true;
handles.fineNumIterations = 1;
handles.fineStripHeight = 15;
handles.fineStripWidth = 488;
handles.fineSamplingRate = 540;
handles.fineMinPeakRatio = 0.8;
handles.fineMinPeakThreshold = 0.2;
handles.fineAdaptiveSearch = false;
handles.fineScalingFactor = 8;
handles.fineSearchWindowHeight = 79;
handles.fineSubpixelInterp = true;
handles.fineNeighborhoodSize = 7;
handles.fineSubpixelDepth = 2;
% Strip
handles.stripOverwrite = true;
handles.stripVerbosity = true;
handles.stripStripHeight = 15;
handles.stripStripWidth = 488;
handles.stripSamplingRate = 540;
handles.stripEnableGaussFilt = true;
handles.stripDisableGaussFilt = false;
handles.stripGaussSD = 10;
handles.stripMinPeakRatio = 0.8;
handles.stripMinPeakThreshold = 0;
handles.stripAdaptiveSearch = false;
handles.stripScalingFactor = 8;
handles.stripSearchWindowHeight = 79;
handles.stripSubpixelInterp = true;
handles.stripNeighborhoodSize = 7;
handles.stripSubpixelDepth = 2;
% Sac
handles.sacOverwrite = true;
handles.sacVerbosity = true;
handles.sacThresholdVal = 6;
handles.sacSecThresholdVal = 2;
handles.sacStitch = 15;
handles.sacMinAmplitude = 0.1;
handles.sacMaxDuration = 100;
handles.sacDetectionMethod1 = false;
handles.sacHardVelThreshold = 35;
handles.sacHardSecondaryVelThreshold = 35;
handles.sacDetectionMethod2 = true;
handles.sacVelMethod1 = true;
handles.sacVelMethod2 = false;

% Pre-Disabled Toggle Values
handles.preDisabledTogTrimValue = 1;
handles.preDisabledTogStimValue = 1;
handles.preDisabledTogGammaValue = 1;
handles.preDisabledTogBandFiltValue = 1;

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

% --- Executes on button press in delete.
function delete_Callback(hObject, eventdata, handles)
% hObject    handle to delete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in moveUp.
function moveUp_Callback(hObject, eventdata, handles)
% hObject    handle to moveUp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in moveDown.
function moveDown_Callback(hObject, eventdata, handles)
% hObject    handle to moveDown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in duplicate.
function duplicate_Callback(hObject, eventdata, handles)
% hObject    handle to duplicate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in edit.
function edit_Callback(hObject, eventdata, handles)
% hObject    handle to edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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
handles.text1.Visible = 'off';
handles.text2.Visible = 'off';
handles.text3.Visible = 'off';
handles.text4.Visible = 'off';
handles.text5.Visible = 'off';
handles.text6.Visible = 'off';
handles.text7.Visible = 'off';
handles.text8.Visible = 'off';
handles.text9.Visible = 'off';
handles.text10.Visible = 'off';
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

handles.files
% Apply modules to all selected files
for i = 1:size(handles.files, 2)
    handles.togTrim.Enable
    if handles.togTrim.Value == 1
        % Set the parameters
        parametersStructure.borderTrimAmount = handles.trimBorderTrimAmount;
        parametersStructure.overwrite = handles.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_dwt' handles.files{i}(end-3:end)];
    end
    
    if handles.togStim.Value == 1
        % Set the parameters
        parametersStructure.enableVerbosity = handles.stimVerbosity;
        parametersStructure.overwrite = handles.stimOverwrite;
        stimulus.thickness = 1; % TODO
        stimulus.size = 51; % TODO
        parametersStructure.thresholdValue = 4; % TODO (used for blink detection)
        
        % Call the function(s)
        FindBlinkFrames(handles.files{i}, parametersStructure);
        FindStimulusLocations(handles.files{i}, stimulus, parametersStructure);
        RemoveStimuli(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_nostim' handles.files{i}(end-3:end)];
    end
    
    if handles.togGamma.Value == 1
        % Set the parameters
        parametersStructure.gammaExponent = handles.gammaExponent;
        parametersStructure.overwrite = handles.gammaOverwrite;
        
        % Call the function(s)
        GammaCorrect(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_gamscaled' handles.files{i}(end-3:end)];
    end
    
    if handles.togBandFilt.Value == 1
        % Set the parameters
        parametersStructure.smoothing = handles.bandFiltSmoothing;
        parametersStructure.lowSpatialFrequencyCutoff = handles.bandFiltFreqCut;
        parametersStructure.overwrite = handles.bandFiltOverwrite;
        
        % Call the function(s)
        BandpassFilter(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_bandfilt' handles.files{i}(end-3:end)];
    end
    
    if handles.togCoarse.Value == 1 % TODO
        % Set the parameters
        parametersStructure.refFrameNumber = handles.coarseRefFrameNum;
        scalingFactor = handles.coarseScalingFactor;
        parametersStructure.overwrite = handles.coarseOverwrite;
        parametersStructure.enableVerbosity = handles.coarseVerbosity;
        parametersStructure.fileName = handles.files{i};
        parametersStructure.enableGPU = false; % TODO

        % Call the function(s)
        coarseResult = CoarseRef(parametersStructure, scalingFactor);
        
        % Update file name to output file name
        %handles.files{i} = [handles.files{i}(1:end-4) '_dwt' handles.files{i}(end-3:end)];
    end
    
    if handles.togFine.Value == 1
        % Set the parameters
        parametersStructure.enableVerbosity = handles.fineVerbosity;
        parametersStructure.numberOfIterations = handles.fineNumIterations;
        parametersStructure.stripHeight = handles.fineStripHeight;
        parametersStructure.stripWidth = handles.fineStripWidth;
        parametersStructure.samplingRate = handles.fineSamplingRate;
        parametersStructure.minimumPeakRatio = handles.fineMinPeakRatio;
        parametersStructure.minimumPeakThreshold = handles.fineMinPeakThreshold;
        parametersStructure.adaptiveSearch = handles.fineAdaptiveSearch;
        parametersStructure.adaptiveSearchScalingFactor = handles.fineScalingFactor;
        parametersStructure.searchWindowHeight = handles.fineSearchWindowHeight;
        parametersStructure.enableSubpixelInterpolation = handles.fineSubpixelInterp;
        parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
            = handles.fineNeighborhoodSize;
        parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
            = handles.fineSubpixelDepth;
        parametersStructure.enableGaussianFiltering = false; % TODO
        parametersStructure.badFrames = []; % TODO
        parametersStructure.axesHandles = []; % TODO        
        
        % Call the function(s)
        fineResult = RefineReferenceFrame(coarseResult, parametersStructure);
        
        % Update file name to output file name
        %handles.files{i} = [handles.files{i}(1:end-4) '_dwt' handles.files{i}(end-3:end)];
    end
    
    if handles.togStrip.Value == 1
        % Set the parameters
        parametersStructure.overwrite = handles.stripOverwrite;
        parametersStructure.enableVerbosity = handles.stripVerbosity;
        parametersStructure.stripHeight = handles.stripStripHeight;
        parametersStructure.stripWidth = handles.stripStripWidth;
        parametersStructure.samplingRate = handles.stripSamplingRate;
        parametersStructure.enableGaussianFiltering = handles.stripEnableGaussFilt;
        parametersStructure.gaussianStandardDeviation = handles.stripGaussSD;
        parametersStructure.minimumPeakRatio = handles.stripMinPeakRatio;
        parametersStructure.minimumPeakThreshold = handles.stripMinPeakThreshold;
        parametersStructure.adaptiveSearch = handles.stripAdaptiveSearch;
        parametersStructure.adaptiveSearchScalingFactor = handles.stripScalingFactor;
        parametersStructure.searchWindowHeight = handles.stripSearchWindowHeight;
        parametersStructure.enableSubpixelInterpolation = handles.stripSubpixelInterp;
        parametersStructure.subpixelInterpolationParameters.neighborhoodSize ...
            = handles.stripNeighborhoodSize;
        parametersStructure.subpixelInterpolationParameters.subpixelDepth ...
            = handles.stripSubpixelDepth;

        % Call the function(s)
        if strcmp(handles.togFine.Enable, 'on') 
            [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
                statisticsStructure] ...
                = StripAnalysis(handles.files{i}, fineResult, parametersStructure);
        elseif strcmp(handles.togCoarse.Enable, 'on')
            [rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
                statisticsStructure] ...
                = StripAnalysis(handles.files{i}, coarseResult, parametersStructure);
        else
            % TODO use a specific frame of the video as reference
        end        
    end
    
    if false
    %if handles.togFilt.Value == 1 % TODO
        % Set the parameters
        parametersStructure.borderTrimAmount = handles.trimBorderTrimAmount;
        parametersStructure.overwrite = handles.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_dwt' handles.files{i}(end-3:end)];
    end
    
    if false
    %if handles.togReRef.Value == 1 % TODO
        % Set the parameters
        parametersStructure.borderTrimAmount = handles.trimBorderTrimAmount;
        parametersStructure.overwrite = handles.trimOverwrite;
        
        % Call the function(s)
        TrimVideo(handles.files{i}, parametersStructure);
        
        % Update file name to output file name
        handles.files{i} = [handles.files{i}(1:end-4) '_dwt' handles.files{i}(end-3:end)];
    end
    
    if handles.togSacDrift.Value == 1
        % Set the parameters
        parametersStructure.overwrite = handles.sacOverwrite;
        parametersStructure.enableVerbosity = handles.sacVerbosity;
        parametersStructure.thresholdValue = handles.sacThresholdVal;
        parametersStructure.secondaryThresholdValue = handles.sacSecThresholdVal;
        parametersStructure.stitchCriteria = handles.sacStitch;
        parametersStructure.minAmplitude = handles.sacMinAmplitude;
        parametersStructure.maxDuration = handles.sacMaxDuration;
        if handles.sacDetectionMethod1
            parametersStructure.detectionMethod = 1;
        else
            parametersStructure.detectionMethod = 2;
        end
        parametersStructure.hardVelocityThreshold = handles.sacHardVelThreshold;
        parametersStructure.hardSecondaryVelocityThreshold = ...
            handles.sacHardSecondaryVelThreshold;
        if handles.sacVelMethod1
            parametersStructure.velocityMethod = 1;
        else
            parametersStructure.velocityMethod = 2;
        end
        
        % Update file name to input file name
        inputFileName = [handles.files{i}(1:end-4) '_' ...
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
    handles.togTrim.Value = handles.preDisabledTogTrimValue;
    handles.togTrim.Enable = 'on';
    handles.configTrim.Enable = 'on';
    togTrim_Callback(handles.togTrim, eventdata, handles);
end

if strcmp(handles.togStim.Enable, 'off')
    handles.togStim.Value = handles.preDisabledTogStimValue;
    handles.togStim.Enable = 'on';
    handles.configStim.Enable = 'on';
    togStim_Callback(handles.togStim, eventdata, handles);
end

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.preDisabledTogBandFiltValue;
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
            if contains(folderFiles(j).name, suffix)
                handles.files = ...
                    [{fullfile(folderFiles(j).folder, folderFiles(j).name)}, ...
                    handles.files];
            end
        end
    elseif ~contains(handles.files(i), suffix)
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
    handles.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'on')
    handles.preDisabledTogGammaValue = handles.togGamma.Value;
end
handles.togGamma.Value = 0;
handles.togGamma.Enable = 'off';
handles.configGamma.Enable = 'off';
togGamma_Callback(handles.togGamma, eventdata, handles);

if strcmp(handles.togBandFilt.Enable, 'on')
    handles.preDisabledTogBandFiltValue = handles.togBandFilt.Value;
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
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
end

% --- Executes on button press in togStrip.
function togStrip_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = [.49 .18 .56];
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
    hObject.BackgroundColor = [.69 .49 .74];
end

% --- Executes on button press in togStim.
function togStim_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
end

% --- Executes on button press in togGamma.
function togGamma_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
end

% --- Executes on button press in togCoarse.
function togCoarse_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = [.49 .18 .56];
else
    if handles.togFine.Value == 1
        errordlg(...
            'Make Coarse Reference Frame must be enabled if Make Fine Reference Frame is enabled.', 'Invalid Selection');
        hObject.Value = 1;
        return;
    end
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
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
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
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
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
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
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
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
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
end

% --- Executes on button press in togBandFilt.
function togBandFilt_Callback(hObject, eventdata, handles)
if hObject.Value == 1
    hObject.String = 'ENABLED';
    hObject.BackgroundColor = [.49 .18 .56];
else
    hObject.String = 'DISABLED';
    hObject.BackgroundColor = [.69 .49 .74];
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
    handles.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'off')
    handles.togStim.Value = handles.preDisabledTogStimValue;
    handles.togStim.Enable = 'on';
    handles.configStim.Enable = 'on';
    togStim_Callback(handles.togStim, eventdata, handles);
end

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.preDisabledTogBandFiltValue;
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
    handles.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'off')
    handles.togGamma.Value = handles.preDisabledTogGammaValue;
    handles.togGamma.Enable = 'on';
    handles.configGamma.Enable = 'on';
    togGamma_Callback(handles.togGamma, eventdata, handles);
end

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.preDisabledTogBandFiltValue;
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
    handles.preDisabledTogTrimValue = handles.togTrim.Value;
end
handles.togTrim.Value = 0;
handles.togTrim.Enable = 'off';
handles.configTrim.Enable = 'off';
togTrim_Callback(handles.togTrim, eventdata, handles);

if strcmp(handles.togStim.Enable, 'on')
    handles.preDisabledTogStimValue = handles.togStim.Value;
end
handles.togStim.Value = 0;
handles.togStim.Enable = 'off';
handles.configStim.Enable = 'off';
togStim_Callback(handles.togStim, eventdata, handles);

if strcmp(handles.togGamma.Enable, 'on')
    handles.preDisabledTogGammaValue = handles.togGamma.Value;
end
handles.togGamma.Value = 0;
handles.togGamma.Enable = 'off';
handles.configGamma.Enable = 'off';
togGamma_Callback(handles.togGamma, eventdata, handles);

if strcmp(handles.togBandFilt.Enable, 'off')
    handles.togBandFilt.Value = handles.preDisabledTogBandFiltValue;
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
