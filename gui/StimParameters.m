function varargout = StimParameters(varargin)
% STIMPARAMETERS MATLAB code for StimParameters.fig
%      STIMPARAMETERS, by itself, creates a new STIMPARAMETERS or raises the existing
%      singleton*.
%
%      H = STIMPARAMETERS returns the handle to a new STIMPARAMETERS or the handle to
%      the existing singleton*.
%
%      STIMPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STIMPARAMETERS.M with the given input arguments.
%
%      STIMPARAMETERS('Property','Value',...) creates a new STIMPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before StimParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to StimParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help StimParameters

% Last Modified by GUIDE v2.5 26-Jun-2017 13:04:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @StimParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @StimParameters_OutputFcn, ...
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


% --- Executes just before StimParameters is made visible.
function StimParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to StimParameters (see VARARGIN)

% Choose default command line output for StimParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.verbosity.Value = mainHandles.stimVerbosity;
handles.overwrite.Value = mainHandles.stimOverwrite;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes StimParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = StimParameters_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in overwrite.
function overwrite_Callback(hObject, eventdata, handles)
% hObject    handle to overwrite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of overwrite


% --- Executes on button press in verbosity.
function verbosity_Callback(hObject, eventdata, handles)
% hObject    handle to verbosity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of verbosity


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

% Save new configurations
mainHandles.stimVerbosity = logical(handles.verbosity.Value);
mainHandles.stimOverwrite = logical(handles.overwrite.Value);

% Update handles structure
guidata(figureHandle, mainHandles);

close;

% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;
