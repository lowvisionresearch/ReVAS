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

% Last Modified by GUIDE v2.5 11-Nov-2017 13:25:51

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

handles.overwrite.Value = mainHandles.config.sacOverwrite;
handles.verbosity.Value = mainHandles.config.sacVerbosity;
handles.thresholdVal.String = mainHandles.config.sacThresholdVal;
handles.secThresholdVal.String = mainHandles.config.sacSecThresholdVal;
handles.stitch.String = mainHandles.config.sacStitch;
handles.minAmplitude.String = mainHandles.config.sacMinAmplitude;
handles.maxDuration.String = mainHandles.config.sacMaxDuration;
handles.minDuration.String = mainHandles.config.sacMinDuration;
handles.detectionMethod1.Value = mainHandles.config.sacDetectionMethod1;
handles.hardVelThreshold.String = mainHandles.config.sacHardVelThreshold;
handles.hardSecondaryVelThreshold.String = mainHandles.config.sacHardSecondaryVelThreshold;
handles.detectionMethod2.Value = mainHandles.config.sacDetectionMethod2;
handles.velMethod1.Value = mainHandles.config.sacVelMethod1;
handles.velMethod2.Value = mainHandles.config.sacVelMethod2;

if logical(handles.detectionMethod1.Value)
    handles.hardVelThreshold.Enable = 'on';
    handles.hardSecondaryVelThreshold.Enable = 'on';
    handles.thresholdVal.Enable = 'off';
    handles.secThresholdVal.Enable = 'off';
else
    handles.hardVelThreshold.Enable = 'off';
    handles.hardSecondaryVelThreshold.Enable = 'off';
    handles.thresholdVal.Enable = 'on';
    handles.secThresholdVal.Enable = 'on';
end

% Set colors
% Main Background
handles.sacParameters.Color = mainHandles.colors{4,2};
handles.thresholdVal.BackgroundColor = mainHandles.colors{4,2};
handles.secThresholdVal.BackgroundColor = mainHandles.colors{4,2};
handles.stitch.BackgroundColor = mainHandles.colors{4,2};
handles.minAmplitude.BackgroundColor = mainHandles.colors{4,2};
handles.maxDuration.BackgroundColor = mainHandles.colors{4,2};
handles.minDuration.BackgroundColor = mainHandles.colors{4,2};
handles.hardVelThreshold.BackgroundColor = mainHandles.colors{4,2};
handles.hardSecondaryVelThreshold.BackgroundColor = mainHandles.colors{4,2};
% Box backgrounds
handles.titleBox.BackgroundColor = mainHandles.colors{4,3};
handles.usageBox.BackgroundColor = mainHandles.colors{4,3};
handles.sacBox.BackgroundColor = mainHandles.colors{4,3};
handles.detectionBox.BackgroundColor = mainHandles.colors{4,3};
handles.velBox.BackgroundColor = mainHandles.colors{4,3};
handles.overwrite.BackgroundColor = mainHandles.colors{4,3};
handles.verbosity.BackgroundColor = mainHandles.colors{4,3};
handles.stitchText.BackgroundColor = mainHandles.colors{4,3};
handles.ampText.BackgroundColor = mainHandles.colors{4,3};
handles.durText.BackgroundColor = mainHandles.colors{4,3};
handles.minDurText.BackgroundColor = mainHandles.colors{4,3};
handles.detectionMethod1.BackgroundColor = mainHandles.colors{4,3};
handles.hardThreshText.BackgroundColor = mainHandles.colors{4,3};
handles.hardThreshTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.hardSecThreshText.BackgroundColor = mainHandles.colors{4,3};
handles.hardSecThreshTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.threshValText.BackgroundColor = mainHandles.colors{4,3};
handles.threshValTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.secThreshValText.BackgroundColor = mainHandles.colors{4,3};
handles.secThreshValTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.detectionMethod2.BackgroundColor = mainHandles.colors{4,3};
handles.detectionGroup.BackgroundColor = mainHandles.colors{4,3};
handles.velMethod1.BackgroundColor = mainHandles.colors{4,3};
handles.velMethod2.BackgroundColor = mainHandles.colors{4,3};
handles.velGroup.BackgroundColor = mainHandles.colors{4,3};
% Box text
handles.titleBox.ForegroundColor = mainHandles.colors{4,5};
handles.usageBox.ForegroundColor = mainHandles.colors{4,5};
handles.sacBox.ForegroundColor = mainHandles.colors{4,5};
handles.detectionBox.ForegroundColor = mainHandles.colors{4,5};
handles.velBox.ForegroundColor = mainHandles.colors{4,5};
handles.overwrite.ForegroundColor = mainHandles.colors{4,5};
handles.verbosity.ForegroundColor = mainHandles.colors{4,5};
handles.stitchText.ForegroundColor = mainHandles.colors{4,5};
handles.ampText.ForegroundColor = mainHandles.colors{4,5};
handles.durText.ForegroundColor = mainHandles.colors{4,5};
handles.minDurText.ForegroundColor = mainHandles.colors{4,5};
handles.detectionMethod1.ForegroundColor = mainHandles.colors{4,5};
handles.hardThreshText.ForegroundColor = mainHandles.colors{4,5};
handles.hardThreshTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.hardSecThreshText.ForegroundColor = mainHandles.colors{4,5};
handles.hardSecThreshTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.threshValText.ForegroundColor = mainHandles.colors{4,5};
handles.threshValTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.secThreshValText.ForegroundColor = mainHandles.colors{4,5};
handles.secThreshValTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.detectionMethod2.ForegroundColor = mainHandles.colors{4,5};
handles.velMethod1.ForegroundColor = mainHandles.colors{4,5};
handles.velMethod2.ForegroundColor = mainHandles.colors{4,5};
handles.thresholdVal.ForegroundColor = mainHandles.colors{4,5};
handles.secThresholdVal.ForegroundColor = mainHandles.colors{4,5};
handles.stitch.ForegroundColor = mainHandles.colors{4,5};
handles.minAmplitude.ForegroundColor = mainHandles.colors{4,5};
handles.maxDuration.ForegroundColor = mainHandles.colors{4,5};
handles.minDuration.ForegroundColor = mainHandles.colors{4,5};
handles.hardVelThreshold.ForegroundColor = mainHandles.colors{4,5};
handles.hardSecondaryVelThreshold.ForegroundColor = mainHandles.colors{4,5};
% Save button
handles.save.BackgroundColor = mainHandles.colors{3,4};
handles.save.ForegroundColor = mainHandles.colors{3,2};
% Cancel button
handles.cancel.BackgroundColor = mainHandles.colors{2,4};
handles.cancel.ForegroundColor = mainHandles.colors{2,2};

% Update handles structure
guidata(hObject, handles);

% Check parameter validity and change colors if needed
hardVelThreshold_Callback(handles.hardVelThreshold, eventdata, handles);
hardSecondaryVelThreshold_Callback(handles.hardSecondaryVelThreshold, eventdata, handles);
thresholdVal_Callback(handles.thresholdVal, eventdata, handles);
secThresholdVal_Callback(handles.secThresholdVal, eventdata, handles);

% UIWAIT makes SacDriftParameters wait for user response (see UIRESUME)
% uiwait(handles.sacParameters);


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
% stitch
stitch = str2double(handles.stitch.String);
if ~IsNaturalNumber(stitch)
    errordlg('Stitch Criteria must be a natural number.', 'Invalid Parameter');
    return;
end

% minAmplitude
minAmplitude = str2double(handles.minAmplitude.String);
if ~IsRealNumber(minAmplitude)
    errordlg('Minimum Amplitude must be a real number.', 'Invalid Parameter');
    return;
end

% maxDuration
maxDuration = str2double(handles.maxDuration.String);
if ~IsPositiveRealNumber(maxDuration)
    errordlg('Maximum Duration must be a positive real number.', 'Invalid Parameter');
    return;
end

% minDuration
minDuration = str2double(handles.minDuration.String);
if ~IsPositiveRealNumber(minDuration)
    errordlg('Minimum Duration must be a positive real number.', 'Invalid Parameter');
    return;
end

if logical(handles.detectionMethod1.Value)
    % hardVelThreshold
    hardVelThreshold = str2double(handles.hardVelThreshold.String);
    if ~IsNonNegativeRealNumber(hardVelThreshold)
        errordlg('Hard Velocity Threshold must be a non-negative, real number.', 'Invalid Parameter');
        return;
    end

    % hardSecondaryVelThreshold
    hardSecondaryVelThreshold = str2double(handles.hardSecondaryVelThreshold.String);
    if ~IsNonNegativeRealNumber(hardSecondaryVelThreshold)
        errordlg('Hard Secondary Velocity Threshold must be a non-negative, real number.', 'Invalid Parameter');
        return;
    end
else
    % thresholdVal
    thresholdVal = str2double(handles.thresholdVal.String);
    if ~IsNonNegativeRealNumber(thresholdVal)
        errordlg('Threshold Value must be a non-negative, real number.', 'Invalid Parameter');
        return;
    end

    % secThresholdVal
    secThresholdVal = str2double(handles.secThresholdVal.String);
    if ~IsNonNegativeRealNumber(secThresholdVal)
        errordlg('Secondary Threshold Value must be a non-negative, real number.', 'Invalid Parameter');
        return;
    end
end

% Save new configurations
mainHandles.config.sacOverwrite = logical(handles.overwrite.Value);
mainHandles.config.sacVerbosity = logical(handles.verbosity.Value);
mainHandles.config.sacStitch = str2double(handles.stitch.String);
mainHandles.config.sacMinAmplitude = str2double(handles.minAmplitude.String);
mainHandles.config.sacMaxDuration = str2double(handles.maxDuration.String);
mainHandles.config.sacMinDuration = str2double(handles.minDuration.String);
mainHandles.config.sacVelMethod1 = logical(handles.velMethod1.Value);
mainHandles.config.sacVelMethod2 = logical(handles.velMethod2.Value);
mainHandles.config.sacDetectionMethod1 = logical(handles.detectionMethod1.Value);
mainHandles.config.sacHardVelThreshold = str2double(handles.hardVelThreshold.String);
mainHandles.config.sacHardSecondaryVelThreshold = ...
    str2double(handles.hardSecondaryVelThreshold.String);
mainHandles.config.sacDetectionMethod2 = logical(handles.detectionMethod2.Value);
mainHandles.config.sacThresholdVal = str2double(handles.thresholdVal.String);
mainHandles.config.sacSecThresholdVal = str2double(handles.secThresholdVal.String);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNonNegativeRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a non-negative, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNonNegativeRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a non-negative, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNonNegativeRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a non-negative, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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
        handles.thresholdVal.Enable = 'off';
        handles.secThresholdVal.Enable = 'off';
    else
        handles.hardVelThreshold.Enable = 'off';
        handles.hardSecondaryVelThreshold.Enable = 'off';
        handles.thresholdVal.Enable = 'on';
        handles.secThresholdVal.Enable = 'on';
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
        handles.thresholdVal.Enable = 'on';
        handles.secThresholdVal.Enable = 'on';
    else
        handles.hardVelThreshold.Enable = 'on';
        handles.hardSecondaryVelThreshold.Enable = 'on';
        handles.thresholdVal.Enable = 'off';
        handles.secThresholdVal.Enable = 'off';
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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNonNegativeRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a non-negative, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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

function minDuration_Callback(hObject, eventdata, handles)
% hObject    handle to minDuration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minDuration as text
%        str2double(get(hObject,'String')) returns contents of minDuration as a double
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function minDuration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minDuration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
