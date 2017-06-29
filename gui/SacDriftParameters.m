function varargout = SacDriftParameters(varargin)
% SACDRIFTPARAMETERS MATLAB code for SacDriftParameters.fig
%      SACDRIFTPARAMETERS, by itself, creates a new SACDRIFTPARAMETERS or raises the existing
%      singleton*.
%
%      H = SACDRIFTPARAMETERS returns the handle to a new SACDRIFTPARAMETERS or the handle to
%      the existing singleton*.
%
%      SACDRIFTPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SACDRIFTPARAMETERS.M with the given input arguments.
%
%      SACDRIFTPARAMETERS('Property','Value',...) creates a new SACDRIFTPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SacDriftParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SacDriftParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SacDriftParameters

% Last Modified by GUIDE v2.5 29-Jun-2017 15:38:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SacDriftParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @SacDriftParameters_OutputFcn, ...
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


% --- Executes just before SacDriftParameters is made visible.
function SacDriftParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SacDriftParameters (see VARARGIN)

% Choose default command line output for SacDriftParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.overwrite.Value = mainHandles.sacOverwrite;
handles.verbosity.Value = mainHandles.sacVerbosity;
handles.thresholdVal.String = mainHandles.sacThresholdVal;
handles.secThresholdVal.String = mainHandles.sacSecThresholdVal;
handles.stitch.String = mainHandles.sacStitch;
handles.minAmplitude.String = mainHandles.sacMinAmplitude;
handles.maxDuration.String = mainHandles.sacMaxDuration;
handles.detectionMethod1.Value = mainHandles.sacDetectionMethod1;
handles.hardVelThreshold.String = mainHandles.sacHardVelThreshold;
handles.hardSecondaryVelThreshold.String = mainHandles.sacHardSecondaryVelThreshold;
handles.detectionMethod2.Value = mainHandles.sacDetectionMethod2;
handles.velMethod1.Value = mainHandles.sacVelMethod1;
handles.velMethod2.Value = mainHandles.sacVelMethod2;

if logical(handles.detectionMethod1.Value)
    handles.hardVelThreshold.Enable = 'on';
    handles.hardSecondaryVelThreshold.Enable = 'on';
else
    handles.hardVelThreshold.Enable = 'off';
    handles.hardSecondaryVelThreshold.Enable = 'off';
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SacDriftParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SacDriftParameters_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

% Validate new configurations
% thresholdVal
thresholdVal = str2double(handles.thresholdVal.String);
if isnan(thresholdVal)
    errordlg('Threshold Value must be a real number', 'Invalid Parameter');
    return;
end

% secThresholdVal
secThresholdVal = str2double(handles.secThresholdVal.String);
if isnan(secThresholdVal)
    errordlg('Secondary Threshold Value must be a real number', 'Invalid Parameter');
    return;
end

% stitch
stitch = str2double(handles.stitch.String);
if isnan(stitch) || ...
    stitch < 0 || ...
    rem(stitch,1) ~= 0
    errordlg('Stitch Criteria must be a natural number', 'Invalid Parameter');
    return;
end

% minAmplitude
minAmplitude = str2double(handles.minAmplitude.String);
if isnan(minAmplitude)
    errordlg('Minimum Amplitude must be a real number', 'Invalid Parameter');
    return;
end

% maxDuration
maxDuration = str2double(handles.maxDuration.String);
if isnan(maxDuration) || ...
        maxDuration < 0
    errordlg('Maximum Duration must be a positive real number', 'Invalid Parameter');
    return;
end

% hardVelThreshold
hardVelThreshold = str2double(handles.hardVelThreshold.String);
if isnan(hardVelThreshold)
    errordlg('Hard Velocity Threshold must be a real number', 'Invalid Parameter');
    return;
end

% hardSecondaryVelThreshold
hardSecondaryVelThreshold = str2double(handles.hardSecondaryVelThreshold.String);
if isnan(hardSecondaryVelThreshold)
    errordlg('Hard Secondary Velocity Threshold must be a real number', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.sacOverwrite = logical(handles.overwrite.Value);
mainHandles.sacVerbosity = logical(handles.verbosity.Value);
mainHandles.sacThresholdVal = str2double(handles.thresholdVal.String);
mainHandles.sacSecThresholdVal = str2double(handles.secThresholdVal.String);
mainHandles.sacStitch = str2double(handles.stitch.String);
mainHandles.sacMinAmplitude = str2double(handles.minAmplitude.String);
mainHandles.sacMaxDuration = str2double(handles.maxDuration.String);
mainHandles.sacDetectionMethod1 = logical(handles.detectionMethod1.Value);
mainHandles.sacHardVelThreshold = str2double(handles.hardVelThreshold.String);
mainHandles.sacHardSecondaryVelThreshold = ...
    str2double(handles.hardSecondaryVelThreshold.String);
mainHandles.sacDetectionMethod2 = logical(handles.detectionMethod2.Value);
mainHandles.sacVelMethod1 = logical(handles.velMethod1.Value);
mainHandles.sacVelMethod2 = logical(handles.velMethod2.Value);

% Update handles structure
guidata(figureHandle, mainHandles);

close;


% --- Executes on button press in overwrite.
function overwrite_Callback(hObject, eventdata, handles)
% hObject    handle to overwrite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of overwrite


% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;



function hardVelThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to hardVelThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hardVelThreshold as text
%        str2double(get(hObject,'String')) returns contents of hardVelThreshold as a double


% --- Executes during object creation, after setting all properties.
function hardVelThreshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hardVelThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in verbosity.
function verbosity_Callback(hObject, eventdata, handles)
% hObject    handle to verbosity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of verbosity


function thresholdVal_Callback(hObject, eventdata, handles)
% hObject    handle to thresholdVal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of thresholdVal as text
%        str2double(get(hObject,'String')) returns contents of thresholdVal as a double


% --- Executes during object creation, after setting all properties.
function thresholdVal_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thresholdVal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function secThresholdVal_Callback(hObject, eventdata, handles)
% hObject    handle to secThresholdVal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of secThresholdVal as text
%        str2double(get(hObject,'String')) returns contents of secThresholdVal as a double


% --- Executes during object creation, after setting all properties.
function secThresholdVal_CreateFcn(hObject, eventdata, handles)
% hObject    handle to secThresholdVal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stitch_Callback(hObject, eventdata, handles)
% hObject    handle to stitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stitch as text
%        str2double(get(hObject,'String')) returns contents of stitch as a double


% --- Executes during object creation, after setting all properties.
function stitch_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function minAmplitude_Callback(hObject, eventdata, handles)
% hObject    handle to minAmplitude (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minAmplitude as text
%        str2double(get(hObject,'String')) returns contents of minAmplitude as a double


% --- Executes during object creation, after setting all properties.
function minAmplitude_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minAmplitude (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxDuration_Callback(hObject, eventdata, handles)
% hObject    handle to maxDuration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxDuration as text
%        str2double(get(hObject,'String')) returns contents of maxDuration as a double


% --- Executes during object creation, after setting all properties.
function maxDuration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxDuration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in detectionMethod1.
function detectionMethod1_Callback(hObject, eventdata, handles)
% hObject    handle to detectionMethod1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of detectionMethod1
    if get(hObject,'Value') == 1
        handles.hardVelThreshold.Enable = 'on';
        handles.hardSecondaryVelThreshold.Enable = 'on';
    else
        handles.hardVelThreshold.Enable = 'off';
        handles.hardSecondaryVelThreshold.Enable = 'off';
    end

% --- Executes on button press in detectionMethod2.
function detectionMethod2_Callback(hObject, eventdata, handles)
% hObject    handle to detectionMethod2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of detectionMethod2
    if get(hObject,'Value') == 1
        handles.hardVelThreshold.Enable = 'off';
        handles.hardSecondaryVelThreshold.Enable = 'off';
    else
        handles.hardVelThreshold.Enable = 'on';
        handles.hardSecondaryVelThreshold.Enable = 'on';
    end

% --- Executes on button press in velMethod1.
function velMethod1_Callback(hObject, eventdata, handles)
% hObject    handle to velMethod1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of velMethod1


% --- Executes on button press in velMethod2.
function velMethod2_Callback(hObject, eventdata, handles)
% hObject    handle to velMethod2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of velMethod2



function hardSecondaryVelThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to hardSecondaryVelThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hardSecondaryVelThreshold as text
%        str2double(get(hObject,'String')) returns contents of hardSecondaryVelThreshold as a double


% --- Executes during object creation, after setting all properties.
function hardSecondaryVelThreshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hardSecondaryVelThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
