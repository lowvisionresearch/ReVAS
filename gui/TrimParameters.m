function varargout = TrimParameters(varargin)
% TRIMPARAMETERS MATLAB code for TrimParameters.fig
%      TRIMPARAMETERS, by itself, creates a new TRIMPARAMETERS or raises the existing
%      singleton*.
%
%      H = TRIMPARAMETERS returns the handle to a new TRIMPARAMETERS or the handle to
%      the existing singleton*.
%
%      TRIMPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRIMPARAMETERS.M with the given input arguments.
%
%      TRIMPARAMETERS('Property','Value',...) creates a new TRIMPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TrimParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TrimParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TrimParameters

% Last Modified by GUIDE v2.5 16-Jun-2017 12:55:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TrimParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @TrimParameters_OutputFcn, ...
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


% --- Executes just before TrimParameters is made visible.
function TrimParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TrimParameters (see VARARGIN)

% Choose default command line output for TrimParameters
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TrimParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = TrimParameters_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in browse1.
function browse1_Callback(hObject, eventdata, handles)
% hObject    handle to browse1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file, folder] = uigetfile('*.*');
path = fullfile(folder, file);
set(handles.path1,'String',path);


% --- Executes on button press in add.
function add_Callback(hObject, eventdata, handles)
% hObject    handle to add (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figureHandle = findobj(0, 'tag', 'jobQueue');
jobQueueHandles = guidata(figureHandle);

% Populate table
jobQueueHandles.table.Data{1,1} = 'Pre-01: Trim';

% Extract the file name from full path before displaying in the table
path = get(handles.path1, 'String');
index = strfind(path, '\');
if isempty(index)
    index = 1;
else
    index = index(end)+1;
end
jobQueueHandles.table.Data{1,2} = path(index:end);

jobQueueHandles.table.Data{1,3} = [path(index:end-4) '_dwt' path(end-3:end)];

% Populate parameters into hiddenTable
jobQueueHandles.hiddenTable.Data{1,1} = handles.path1.String;
jobQueueHandles.hiddenTable.Data{1,2} = handles.borderTrimAmount.String;
jobQueueHandles.hiddenTable.Data{1,3} = handles.overwrite.Value;

close;


% --- Executes on button press in overwrite.
function overwrite_Callback(hObject, eventdata, handles)
% hObject    handle to overwrite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of overwrite



function borderTrimAmount_Callback(hObject, eventdata, handles)
% hObject    handle to borderTrimAmount (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of borderTrimAmount as text
%        str2double(get(hObject,'String')) returns contents of borderTrimAmount as a double


% --- Executes during object creation, after setting all properties.
function borderTrimAmount_CreateFcn(hObject, eventdata, handles)
% hObject    handle to borderTrimAmount (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;
