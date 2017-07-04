function varargout = Parallelization(varargin)
% PARALLELIZATION MATLAB code for Parallelization.fig
%      PARALLELIZATION, by itself, creates a new PARALLELIZATION or raises the existing
%      singleton*.
%
%      H = PARALLELIZATION returns the handle to a new PARALLELIZATION or the handle to
%      the existing singleton*.
%
%      PARALLELIZATION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PARALLELIZATION.M with the given input arguments.
%
%      PARALLELIZATION('Property','Value',...) creates a new PARALLELIZATION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Parallelization_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Parallelization_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Parallelization

% Last Modified by GUIDE v2.5 30-Jun-2017 17:02:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Parallelization_OpeningFcn, ...
                   'gui_OutputFcn',  @Parallelization_OutputFcn, ...
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


% --- Executes just before Parallelization is made visible.
function Parallelization_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Parallelization (see VARARGIN)

% Choose default command line output for Parallelization
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.enableMultiCore.Value = mainHandles.config.parMultiCore;
handles.enableGPU.Value = mainHandles.config.parGPU;

% Set colors
% Main Background
handles.parallelization.Color = mainHandles.colors{4,2};
% Box backgrounds
handles.titleBox.BackgroundColor = mainHandles.colors{4,3};
handles.parallelizationBox.BackgroundColor = mainHandles.colors{4,3};
handles.enableMultiCore.BackgroundColor = mainHandles.colors{4,3};
handles.enableGPU.BackgroundColor = mainHandles.colors{4,3};
% Box text
handles.titleBox.ForegroundColor = mainHandles.colors{4,5};
handles.parallelizationBox.ForegroundColor = mainHandles.colors{4,5};
handles.enableMultiCore.ForegroundColor = mainHandles.colors{4,5};
handles.enableGPU.ForegroundColor = mainHandles.colors{4,5};
% Save button
handles.save.BackgroundColor = mainHandles.colors{3,4};
handles.save.ForegroundColor = mainHandles.colors{3,2};
% Cancel button
handles.cancel.BackgroundColor = mainHandles.colors{2,4};
handles.cancel.ForegroundColor = mainHandles.colors{2,2};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Parallelization wait for user response (see UIRESUME)
% uiwait(handles.parallelization);


% --- Outputs from this function are returned to the command line.
function varargout = Parallelization_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in enableMultiCore.
function enableMultiCore_Callback(hObject, eventdata, handles)
% hObject    handle to enableMultiCore (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of enableMultiCore


% --- Executes on button press in enableGPU.
function enableGPU_Callback(hObject, eventdata, handles)
% hObject    handle to enableGPU (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of enableGPU


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

% Save new configurations
mainHandles.config.parMultiCore = logical(handles.enableMultiCore.Value);
mainHandles.config.parGPU = logical(handles.enableGPU.Value);

% Update handles structure
guidata(figureHandle, mainHandles);

close;

% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;
