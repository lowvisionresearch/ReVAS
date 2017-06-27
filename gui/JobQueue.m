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

% Last Modified by GUIDE v2.5 23-Jun-2017 14:51:06

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
handles.fineOverwrite = true;
handles.fineVerbosity = true;
handles.fineStripHeight = 15;
handles.fineStripWidth = 488;
handles.fineSamplingRate = 540;
handles.fineGaussFilt = true;
handles.fineGaussSD = 10;
handles.fineMinPeakRatio = 0.8;
handles.fineMinPeakThreshold = 0;
handles.fineAdaptiveSearch = false;
handles.fineScalingFactor = 8;
handles.fineSearchWindowHeight = 79;
handles.fineSubpixelInterp = true;
handles.fineNeighborhoodSize = 7;
handles.fineSubpixelDepth = 2;
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
handles.sacDetectionMethod2 = true;
handles.sacVelMethod1 = true;
handles.sacVelMethod2 = false;

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

% Saving initial button states
parallelizationState = get(handles.parallelization, 'enable');

% Turning all buttons off
set(hObject, 'enable', 'off');
set(handles.parallelization, 'enable', 'off');
set(handles.add, 'enable', 'off');
set(handles.delete, 'enable', 'off');
set(handles.moveUp, 'enable', 'off');
set(handles.moveDown, 'enable', 'off');
set(handles.duplicate, 'enable', 'off');
set(handles.edit, 'enable', 'off');

for i = 1:size(handles.inputList.Data, 1)-1
    if handles.inputList.Data{i,1} == 'Pre-01: Trim'
        parametersStructure = struct;
        parametersStructure.borderTrimAmount = str2double(handles.hiddenTable.Data{i,2});
        parametersStructure.overwrite = handles.hiddenTable.Data{i,3};
        path = handles.hiddenTable.Data{i,1};
        TrimVideo(path, parametersStructure);
        fprintf('Process Completed\n');
    end
end

% Renable buttons
set(hObject, 'enable', 'on');
if isequal(parallelizationState, 'on')
    set(handles.parallelization, 'enable', 'on');
end
set(handles.add, 'enable', 'on');


% --- Executes on button press in radioRaw.
function radioRaw_Callback(hObject, eventdata, handles)
% hObject    handle to radioRaw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioRaw


% --- Executes on button press in radiobutton7.
function radiobutton7_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton7


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
    if isinteger(handles.files) && handles.files == 0
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

% --- Executes on button press in radioBandFilt.
function radioBandFilt_Callback(hObject, eventdata, handles)
% hObject    handle to radioBandFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radioBandFilt


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
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togBlink.
function togBlink_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togStrip.
function togStrip_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togStim.
function togStim_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togGamma.
function togGamma_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togCoarse.
function togCoarse_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togReRef.
function togReRef_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togFilt.
function togFilt_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togSacDrift.
function togSacDrift_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togFine.
function togFine_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
        hObject.String = 'ENABLED';
        hObject.BackgroundColor = [.49 .18 .56];
    else
        hObject.String = 'DISABLED';
        hObject.BackgroundColor = [.69 .49 .74];
    end

% --- Executes on button press in togBandFilt.
function togBandFilt_Callback(hObject, eventdata, handles)
    if get(hObject,'Value') == 1
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

% --- Executes on button press in configBlink.
function configBlink_Callback(hObject, eventdata, handles)
% hObject    handle to configBlink (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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
