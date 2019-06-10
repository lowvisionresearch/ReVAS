function varargout = GammaParameters(varargin)
% GAMMAPARAMETERS MATLAB code for GammaParameters.fig
%      GAMMAPARAMETERS, by itself, creates a new GAMMAPARAMETERS or raises the existing
%      singleton*.
%
%      H = GAMMAPARAMETERS returns the handle to a new GAMMAPARAMETERS or the handle to
%      the existing singleton*.
%
%      GAMMAPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GAMMAPARAMETERS.M with the given input arguments.
%
%      GAMMAPARAMETERS('Property','Value',...) creates a new GAMMAPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GammaParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GammaParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GammaParameters

% Last Modified by GUIDE v2.5 10-Jun-2019 00:03:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GammaParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @GammaParameters_OutputFcn, ...
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


% --- Executes just before GammaParameters is made visible.
function GammaParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GammaParameters (see VARARGIN)

% Choose default command line output for GammaParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.exponent.String = mainHandles.config.gammaExponent;
handles.overwrite.Value = mainHandles.config.gammaOverwrite;
handles.rbGammaCorrect.Value = mainHandles.config.isGammaCorrect;
handles.rbHisteq.Value = mainHandles.config.isHistEq;


% set a proper size for the main GUI window. a
handles.gammaParameters.Units = 'normalized';
handles.gammaParameters.OuterPosition = mainHandles.GUIposition.gammaParameters;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.gammaParameters);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.gammaParameters.Color = revasColors.background;
handles.exponent.BackgroundColor = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.usageBox.BackgroundColor = revasColors.boxBackground;
handles.gammaBox.BackgroundColor = revasColors.boxBackground;
handles.overwrite.BackgroundColor = revasColors.boxBackground;
handles.exponentText.BackgroundColor = revasColors.boxBackground;
handles.rbHisteq.BackgroundColor = revasColors.boxBackground;
handles.rbGammaCorrect.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.usageBox.ForegroundColor = revasColors.text;
handles.trimBox.ForegroundColor = revasColors.text;
handles.overwrite.ForegroundColor = revasColors.text;
handles.exponentText.ForegroundColor = revasColors.text;
handles.exponent.ForegroundColor = revasColors.text;
handles.rbHisteq.ForegroundColor = revasColors.text;
handles.rbGammaCorrect.ForegroundColor = revasColors.text;
% Save button
handles.save.BackgroundColor = revasColors.pushButtonBackground;
handles.save.ForegroundColor = revasColors.pushButtonText;
% Cancel button
handles.cancel.BackgroundColor = revasColors.pushButtonBackground;
handles.cancel.ForegroundColor = revasColors.pushButtonText;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GammaParameters wait for user response (see UIRESUME)
% uiwait(handles.gammaParameters);


% --- Outputs from this function are returned to the command line.
function varargout = GammaParameters_OutputFcn(hObject, eventdata, handles) 
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
% exponent
exponent = str2double(handles.exponent.String);
if ~IsRealNumber(exponent)
    errordlg('Gamma Exponent must be a real number.', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.config.gammaExponent = str2double(handles.exponent.String);
mainHandles.config.gammaOverwrite = logical(handles.overwrite.Value);
mainHandles.config.isGammaCorrect = logical(handles.rbGammaCorrect.Value); 
mainHandles.config.isHistEq = logical(handles.rbHisteq.Value); 

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


function exponent_Callback(hObject, eventdata, handles)
% hObject    handle to exponent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of exponent as text
%        str2double(get(hObject,'String')) returns contents of exponent as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value) || (value<=0)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end


% --- Executes during object creation, after setting all properties.
function exponent_CreateFcn(hObject, eventdata, handles)
% hObject    handle to exponent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rbGammaCorrect.
function rbGammaCorrect_Callback(hObject, eventdata, handles)
% hObject    handle to rbGammaCorrect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbGammaCorrect
if get(hObject,'Value') == 1
    handles.exponent.Enable = 'on';
else
    handles.exponent.Enable = 'off';
end

% --- Executes on button press in rbHisteq.
function rbHisteq_Callback(hObject, eventdata, handles)
% hObject    handle to rbHisteq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbHisteq


