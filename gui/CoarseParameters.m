function varargout = CoarseParameters(varargin)
% COARSEPARAMETERS MATLAB code for CoarseParameters.fig
%      COARSEPARAMETERS, by itself, creates a new COARSEPARAMETERS or raises the existing
%      singleton*.
%
%      H = COARSEPARAMETERS returns the handle to a new COARSEPARAMETERS or the handle to
%      the existing singleton*.
%
%      COARSEPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in COARSEPARAMETERS.M with the given input arguments.
%
%      COARSEPARAMETERS('Property','Value',...) creates a new COARSEPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CoarseParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CoarseParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CoarseParameters

% Last Modified by GUIDE v2.5 26-Jun-2017 14:15:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CoarseParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @CoarseParameters_OutputFcn, ...
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


% --- Executes just before CoarseParameters is made visible.
function CoarseParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CoarseParameters (see VARARGIN)

% Choose default command line output for CoarseParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.refFrameNum.String = mainHandles.coarseRefFrameNum;
handles.scalingFactor.String = mainHandles.coarseScalingFactor;
handles.overwrite.Value = mainHandles.coarseOverwrite;
handles.verbosity.Value = mainHandles.coarseVerbosity;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CoarseParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CoarseParameters_OutputFcn(hObject, eventdata, handles) 
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
% refFrameNum
refFrameNum = str2double(handles.refFrameNum.String);
if isnan(refFrameNum) || ...
        refFrameNum < 0 || ...
        rem(refFrameNum,1) ~= 0
    errordlg('Reference Frame Number must be a natural number', 'Invalid Parameter');
    return;
end

% scalingFactor
scalingFactor = str2double(handles.scalingFactor.String);
if isnan(scalingFactor) || ...
        scalingFactor < 0
    errordlg('Scaling Factor must be a positive real number', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.coarseRefFrameNum = str2double(handles.refFrameNum.String);
mainHandles.coarseScalingFactor = str2double(handles.scalingFactor.String);
mainHandles.coarseOverwrite = logical(handles.overwrite.Value);
mainHandles.coarseVerbosity = logical(handles.verbosity.Value);

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


% --- Executes on button press in verbosity.
function verbosity_Callback(hObject, eventdata, handles)
% hObject    handle to verbosity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of verbosity



function refFrameNum_Callback(hObject, eventdata, handles)
% hObject    handle to refFrameNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of refFrameNum as text
%        str2double(get(hObject,'String')) returns contents of refFrameNum as a double


% --- Executes during object creation, after setting all properties.
function refFrameNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to refFrameNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function scalingFactor_Callback(hObject, eventdata, handles)
% hObject    handle to scalingFactor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scalingFactor as text
%        str2double(get(hObject,'String')) returns contents of scalingFactor as a double


% --- Executes during object creation, after setting all properties.
function scalingFactor_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scalingFactor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
