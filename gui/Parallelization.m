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
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.enableMultiCore.Value = mainHandles.config.parMultiCore;
handles.enableGPU.Value = mainHandles.config.parGPU;
if ~mainHandles.config.parMultiCore && ~mainHandles.config.parGPU
    handles.noRadio.Value = true;
else
    handles.noRadio.Value = false;
end



% set a proper size for the main GUI window. a
handles.parallelization.Units = 'normalized';
handles.parallelization.OuterPosition = mainHandles.GUIposition.parallelization;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.parallelization);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.parallelization.Color = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.parallelizationBox.BackgroundColor = revasColors.boxBackground;
handles.enableMultiCore.BackgroundColor = revasColors.boxBackground;
handles.enableGPU.BackgroundColor = revasColors.boxBackground;
handles.noRadio.BackgroundColor = revasColors.boxBackground;
handles.radioGroup.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.parallelizationBox.ForegroundColor = revasColors.text;
handles.enableMultiCore.ForegroundColor = revasColors.text;
handles.enableGPU.ForegroundColor = revasColors.text;
handles.noRadio.ForegroundColor = revasColors.text;
handles.radioGroup.BackgroundColor = revasColors.boxBackground;

% Save button
handles.save.BackgroundColor = revasColors.pushButtonBackground;
handles.save.ForegroundColor = revasColors.pushButtonText;
% Cancel button
handles.cancel.BackgroundColor = revasColors.pushButtonBackground;
handles.cancel.ForegroundColor = revasColors.pushButtonText;

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
figureHandle = findobj(0, 'tag', 'revas');
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
