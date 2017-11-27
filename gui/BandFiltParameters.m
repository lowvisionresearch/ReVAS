function varargout = BandFiltParameters(varargin)
% BANDFILTPARAMETERS MATLAB code for BandFiltParameters.fig
%      BANDFILTPARAMETERS, by itself, creates a new BANDFILTPARAMETERS or raises the existing
%      singleton*.
%
%      H = BANDFILTPARAMETERS returns the handle to a new BANDFILTPARAMETERS or the handle to
%      the existing singleton*.
%
%      BANDFILTPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BANDFILTPARAMETERS.M with the given input arguments.
%
%      BANDFILTPARAMETERS('Property','Value',...) creates a new BANDFILTPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BandFiltParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BandFiltParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BandFiltParameters

% Last Modified by GUIDE v2.5 30-Jun-2017 23:44:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BandFiltParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @BandFiltParameters_OutputFcn, ...
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


% --- Executes just before BandFiltParameters is made visible.
function BandFiltParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BandFiltParameters (see VARARGIN)

% Choose default command line output for BandFiltParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.smoothing.String = mainHandles.config.bandFiltSmoothing;
handles.freqCut.String = mainHandles.config.bandFiltFreqCut;
handles.overwrite.Value = mainHandles.config.bandFiltOverwrite;

% Set colors
% Main Background
handles.bandFiltParameters.Color = mainHandles.colors{4,2};
handles.smoothing.BackgroundColor = mainHandles.colors{4,2};
handles.freqCut.BackgroundColor = mainHandles.colors{4,2};
% Box backgrounds
handles.titleBox.BackgroundColor = mainHandles.colors{4,3};
handles.usageBox.BackgroundColor = mainHandles.colors{4,3};
handles.bandFiltBox.BackgroundColor = mainHandles.colors{4,3};
handles.overwrite.BackgroundColor = mainHandles.colors{4,3};
handles.smoothingText.BackgroundColor = mainHandles.colors{4,3};
handles.smoothingTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.freqCutText.BackgroundColor = mainHandles.colors{4,3};
handles.freqCutTextSub.BackgroundColor = mainHandles.colors{4,3};
% Box text
handles.titleBox.ForegroundColor = mainHandles.colors{4,5};
handles.usageBox.ForegroundColor = mainHandles.colors{4,5};
handles.bandFiltBox.ForegroundColor = mainHandles.colors{4,5};
handles.overwrite.ForegroundColor = mainHandles.colors{4,5};
handles.smoothingText.ForegroundColor = mainHandles.colors{4,5};
handles.smoothingTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.freqCutText.ForegroundColor = mainHandles.colors{4,5};
handles.freqCutTextSub.ForegroundColor = mainHandles.colors{4,5};
% Save button
handles.save.BackgroundColor = mainHandles.colors{3,4};
handles.save.ForegroundColor = mainHandles.colors{3,2};
% Cancel button
handles.cancel.BackgroundColor = mainHandles.colors{2,4};
handles.cancel.ForegroundColor = mainHandles.colors{2,2};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BandFiltParameters wait for user response (see UIRESUME)
% uiwait(handles.bandFiltParameters);


% --- Outputs from this function are returned to the command line.
function varargout = BandFiltParameters_OutputFcn(hObject, eventdata, handles) 
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

figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

% Validate new configurations
% smoothing
smoothing = str2double(handles.smoothing.String);
if ~IsNaturalNumber(smoothing)
    errordlg('Smoothing must be a natural number.', 'Invalid Parameter');
    return;
end

% lowSpatialFrequencyCutoff
freqCut = str2double(handles.freqCut.String);
if ~IsPositiveRealNumber(freqCut)
    errordlg('Low Spatial Frequency Cutoff must be a positive real number.', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.config.bandFiltSmoothing = str2double(handles.smoothing.String);
mainHandles.config.bandFiltFreqCut = str2double(handles.freqCut.String);
mainHandles.config.bandFiltOverwrite = logical(handles.overwrite.Value);

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



function smoothing_Callback(hObject, eventdata, handles)
% hObject    handle to smoothing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of smoothing as text
%        str2double(get(hObject,'String')) returns contents of smoothing as a double
figureHandle = findobj(0, 'tag', 'revas');
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


function freqCut_Callback(hObject, eventdata, handles)
% hObject    handle to freqCut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freqCut as text
%        str2double(get(hObject,'String')) returns contents of freqCut as a double
figureHandle = findobj(0, 'tag', 'revas');
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
function smoothing_CreateFcn(hObject, eventdata, handles)
% hObject    handle to smoothing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function freqCut_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freqCut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
