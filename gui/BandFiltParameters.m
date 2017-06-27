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

% Last Modified by GUIDE v2.5 26-Jun-2017 14:00:39

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.smoothing.String = mainHandles.bandFiltSmoothing;
handles.freqCut.String = mainHandles.bandFiltFreqCut;
handles.overwrite.Value = mainHandles.bandFiltOverwrite;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BandFiltParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);


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

figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

% Validate new configurations
% smoothing
smoothing = str2double(handles.smoothing.String);
if isnan(smoothing) || ...
        smoothing < 0 || ...
        rem(smoothing,1) ~= 0
    errordlg('Smoothing must be a natural number', 'Invalid Parameter');
    return;
end

% lowSpatialFrequencyCutoff
freqCut = str2double(handles.freqCut.String);
if isnan(freqCut) || ...
        freqCut < 0
    errordlg('Low Spatial Frequency Cutoff must be a positive real number', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.bandFiltSmoothing = str2double(handles.smoothing.String);
mainHandles.bandFiltFreqCut = str2double(handles.freqCut.String);
mainHandles.bandFiltOverwrite = logical(handles.overwrite.Value);

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



function freqCut_Callback(hObject, eventdata, handles)
% hObject    handle to freqCut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freqCut as text
%        str2double(get(hObject,'String')) returns contents of freqCut as a double


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
