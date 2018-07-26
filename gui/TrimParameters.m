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

% Last Modified by GUIDE v2.5 27-Nov-2017 12:20:45

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

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.left.String = mainHandles.config.trimLeft;
handles.right.String = mainHandles.config.trimRight;
handles.top.String = mainHandles.config.trimTop;
handles.bottom.String = mainHandles.config.trimBottom;
handles.overwrite.Value = mainHandles.config.trimOverwrite;




% set a proper size for the main GUI window. a
handles.trimParameters.Units = 'normalized';
handles.trimParameters.OuterPosition = mainHandles.GUIposition.trimParameters;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.trimParameters);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.trimParameters.Color = revasColors.background;
handles.left.BackgroundColor = revasColors.background;
handles.right.BackgroundColor = revasColors.background;
handles.top.BackgroundColor = revasColors.background;
handles.bottom.BackgroundColor = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.usageBox.BackgroundColor = revasColors.boxBackground;
handles.trimBox.BackgroundColor = revasColors.boxBackground;
handles.overwrite.BackgroundColor = revasColors.boxBackground;
handles.leftText.BackgroundColor = revasColors.boxBackground;
handles.rightText.BackgroundColor = revasColors.boxBackground;
handles.topText.BackgroundColor = revasColors.boxBackground;
handles.bottomText.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.usageBox.ForegroundColor = revasColors.text;
handles.trimBox.ForegroundColor = revasColors.text;
handles.overwrite.ForegroundColor = revasColors.text;
handles.leftText.ForegroundColor = revasColors.text;
handles.left.ForegroundColor = revasColors.text;
handles.rightText.ForegroundColor = revasColors.text;
handles.right.ForegroundColor = revasColors.text;
handles.topText.ForegroundColor = revasColors.text;
handles.top.ForegroundColor = revasColors.text;
handles.bottomText.ForegroundColor = revasColors.text;
handles.bottom.ForegroundColor = revasColors.text;
% Save button
handles.save.BackgroundColor = revasColors.pushButtonBackground;
handles.save.ForegroundColor = revasColors.pushButtonText;
% Cancel button
handles.cancel.BackgroundColor = revasColors.pushButtonBackground;
handles.cancel.ForegroundColor = revasColors.pushButtonText;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TrimParameters wait for user response (see UIRESUME)
% uiwait(handles.trimParameters);

% --- Outputs from this function are returned to the command line.
function varargout = TrimParameters_OutputFcn(hObject, eventdata, handles) 
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
% left
left = str2double(handles.left.String);
if ~IsNaturalNumber(left)
    errordlg('Left Border Trim Amount must be a natural number.', 'Invalid Parameter');
    return;
end

% right
right = str2double(handles.left.String);
if ~IsNaturalNumber(right)
    errordlg('Right Border Trim Amount must be a natural number.', 'Invalid Parameter');
    return;
end

% top
top = str2double(handles.left.String);
if ~IsNaturalNumber(top)
    errordlg('Top Border Trim Amount must be a natural number.', 'Invalid Parameter');
    return;
end

% bottom
bottom = str2double(handles.left.String);
if ~IsNaturalNumber(bottom)
    errordlg('Bottom Border Trim Amount must be a natural number.', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.config.trimLeft = str2double(handles.left.String);
mainHandles.config.trimRight = str2double(handles.right.String);
mainHandles.config.trimTop = str2double(handles.top.String);
mainHandles.config.trimBottom = str2double(handles.bottom.String);
mainHandles.config.trimOverwrite = logical(handles.overwrite.Value);

% Update handles structure
guidata(figureHandle, mainHandles);

close;

% --- Executes on button press in overwrite.
function overwrite_Callback(hObject, eventdata, handles)
% hObject    handle to overwrite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of overwrite

function left_Callback(hObject, eventdata, handles)
% hObject    handle to left (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of left as text
%        str2double(get(hObject,'String')) returns contents of left as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function left_CreateFcn(hObject, eventdata, handles)
% hObject    handle to left (see GCBO)
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

function right_Callback(hObject, eventdata, handles)
% hObject    handle to right (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of right as text
%        str2double(get(hObject,'String')) returns contents of right as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function right_CreateFcn(hObject, eventdata, handles)
% hObject    handle to right (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function top_Callback(hObject, eventdata, handles)
% hObject    handle to top (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of top as text
%        str2double(get(hObject,'String')) returns contents of top as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function top_CreateFcn(hObject, eventdata, handles)
% hObject    handle to top (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function bottom_Callback(hObject, eventdata, handles)
% hObject    handle to bottom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bottom as text
%        str2double(get(hObject,'String')) returns contents of bottom as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function bottom_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bottom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
