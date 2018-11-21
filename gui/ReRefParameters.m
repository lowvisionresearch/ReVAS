function varargout = ReRefParameters(varargin)
% REREFPARAMETERS MATLAB code for ReRefParameters.fig
%      REREFPARAMETERS, by itself, creates a new REREFPARAMETERS or raises the existing
%      singleton*.
%
%      H = REREFPARAMETERS returns the handle to a new REREFPARAMETERS or the handle to
%      the existing singleton*.
%
%      REREFPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REREFPARAMETERS.M with the given input arguments.
%
%      REREFPARAMETERS('Property','Value',...) creates a new REREFPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ReRefParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ReRefParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ReRefParameters

% Last Modified by GUIDE v2.5 15-Nov-2017 11:34:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ReRefParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @ReRefParameters_OutputFcn, ...
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

% --- Executes just before ReRefParameters is made visible.
function ReRefParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ReRefParameters (see VARARGIN)

% Choose default command line output for rerefparameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.verbosity.Value = mainHandles.config.rerefVerbosity;
handles.overwrite.Value = mainHandles.config.rerefOverwrite;
handles.search.String = mainHandles.config.rerefSearch;
handles.peak1Radio.Value = mainHandles.config.rerefPeakMethod == 1;
handles.peak2Radio.Value = mainHandles.config.rerefPeakMethod == 2;
handles.kernel.String = mainHandles.config.rerefKernel;
handles.fixTorsion.Value = mainHandles.config.rerefTorsion;
handles.tiltLow.String = mainHandles.config.rerefTiltLow;
handles.tiltUp.String = mainHandles.config.rerefTiltUp;
handles.tiltStep.String = mainHandles.config.rerefTiltStep;
handles.globalFullPath = mainHandles.config.rerefGlobalFullPath;
handles.globalPath.String = mainHandles.config.rerefGlobalPath;



% set a proper size for the main GUI window. a
handles.rerefParameters.Units = 'normalized';
handles.rerefParameters.OuterPosition = mainHandles.GUIposition.rerefParameters;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.rerefParameters);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.rerefParameters.Color = revasColors.background;
handles.globalPath.BackgroundColor = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.usageBox.BackgroundColor = revasColors.boxBackground;
handles.torsionBox.BackgroundColor = revasColors.boxBackground;
handles.rerefGroup.BackgroundColor = revasColors.boxBackground;
handles.overwrite.BackgroundColor = revasColors.boxBackground;
handles.verbosity.BackgroundColor = revasColors.boxBackground;
handles.peak1Radio.BackgroundColor = revasColors.boxBackground;
handles.peak2Radio.BackgroundColor = revasColors.boxBackground;
handles.kernelText.BackgroundColor = revasColors.boxBackground;
handles.thickText.BackgroundColor = revasColors.boxBackground;
handles.tiltLowText.BackgroundColor = revasColors.boxBackground;
handles.tiltUpText.BackgroundColor = revasColors.boxBackground;
handles.fixTorsion.BackgroundColor = revasColors.boxBackground;
handles.rerefBox.BackgroundColor = revasColors.boxBackground;
handles.searchText.BackgroundColor = revasColors.boxBackground;
handles.peakBox.BackgroundColor = revasColors.boxBackground;
handles.tiltStepText.BackgroundColor = revasColors.boxBackground;
handles.peakGroup.BackgroundColor = revasColors.boxBackground;
handles.globalText.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.usageBox.ForegroundColor = revasColors.text;
handles.torsionBox.ForegroundColor = revasColors.text;
handles.overwrite.ForegroundColor = revasColors.text;
handles.verbosity.ForegroundColor = revasColors.text;
handles.peak1Radio.ForegroundColor = revasColors.text;
handles.peak2Radio.ForegroundColor = revasColors.text;
handles.kernelText.ForegroundColor = revasColors.text;
handles.thickText.ForegroundColor = revasColors.text;
handles.tiltLowText.ForegroundColor = revasColors.text;
handles.tiltUpText.ForegroundColor = revasColors.text;
handles.fixTorsion.ForegroundColor = revasColors.text;
handles.rerefPath.ForegroundColor = revasColors.text;
handles.rerefBox.ForegroundColor = revasColors.text;
handles.searchText.ForegroundColor = revasColors.text;
handles.peakBox.ForegroundColor = revasColors.text;
handles.tiltStepText.ForegroundColor = revasColors.text;
handles.peakGroup.ForegroundColor = revasColors.text;
handles.globalText.ForegroundColor = revasColors.text;
handles.globalPath.ForegroundColor = revasColors.text;
% Select button
handles.select.BackgroundColor = revasColors.pushButtonBackground;
handles.select.ForegroundColor = revasColors.pushButtonText;
% Save button
handles.save.BackgroundColor = revasColors.pushButtonBackground;
handles.save.ForegroundColor = revasColors.pushButtonText;
% Cancel button
handles.cancel.BackgroundColor = revasColors.pushButtonBackground;
handles.cancel.ForegroundColor = revasColors.pushButtonText;

% Update handles structure
guidata(hObject, handles);

% Check parameter validity and change colors if needed
search_Callback(handles.search, eventdata, handles);
kernel_Callback(handles.kernel, eventdata, handles);
tiltLow_Callback(handles.tiltLow, eventdata, handles);
tiltUp_Callback(handles.tiltUp, eventdata, handles);
tiltStep_Callback(handles.tiltStep, eventdata, handles);
peak1Radio_Callback(handles.peak1Radio, eventdata, handles);
peak2Radio_Callback(handles.peak2Radio, eventdata, handles);
fixTorsion_Callback(handles.fixTorsion, eventdata, handles);
globalPath_Callback(handles.globalPath, eventdata, handles);

% UIWAIT makes ReRefParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = ReRefParameters_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function tiltStep_Callback(hObject, eventdata, handles)
% hObject    handle to tiltStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tiltStep as text
%        str2double(get(hObject,'String')) returns contents of tiltStep as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function tiltStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tiltStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function search_Callback(hObject, eventdata, handles)
% hObject    handle to search (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of search as text
%        str2double(get(hObject,'String')) returns contents of search as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value) || value < 0 || value > 1
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a real number between 0 and 1 (inclusive).';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function search_CreateFcn(hObject, eventdata, handles)
% hObject    handle to search (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

% Validate new configurations
search = str2double(handles.search.String);
kernel = str2double(handles.kernel.String);
tiltLow = str2double(handles.tiltLow.String);
tiltUp = str2double(handles.tiltUp.String);
tiltStep = str2double(handles.tiltStep.String);

% Global Reference Frame
if ~IsImageFile(handles.globalFullPath) && ...
        (length(handles.globalFullPath) < 5 || ~strcmp(handles.globalFullPath(end-3:end), '.mat'))
    errordlg('Global Reference Frame must be a mat or image file.', 'Invalid Parameter');
    return;
elseif size(handles.globalFullPath, 2) > 3 && strcmp(handles.globalFullPath(end-3:end), '.mat') && ...
        ismember('globalRef', who('-file', handles.globalFullPath))
    errordlg('Global Reference Frame mat file must contain variable called globalRef.', 'Invalid Parameter');
    return;
end

% search
if ~IsRealNumber(search) || search < 0 || search > 1
    errordlg('Search Zone must be a real number between 0 and 1 (inclusive).', 'Invalid Parameter');
    return;
end

if logical(handles.peak2Radio.Value)
    % kernel
    if ~IsOddNaturalNumber(kernel)
        errordlg('Kernel Size must be an odd, natural number.', 'Invalid Parameter');
        return;
    end
end

if logical(handles.fixTorsion.Value)
    % tiltLow
    if ~IsRealNumber(tiltLow)
        errordlg('Tilt Lower Bound must be a real number.', 'Invalid Parameter');
        return;
    end
    
    % tiltUp
    if ~IsRealNumber(tiltUp)
        errordlg('Tilt Upper Bound must be a real number.', 'Invalid Parameter');
        return;
    end
    
    % tiltStep
    if ~IsRealNumber(tiltStep)
        errordlg('Tilt Step Size must be a real number.', 'Invalid Parameter');
        return;
    end
end

% Save new configurations
mainHandles.config.rerefGlobalFullPath = handles.globalFullPath;
mainHandles.config.rerefGlobalPath = handles.globalPath.String;
mainHandles.config.rerefVerbosity = logical(handles.verbosity.Value);
mainHandles.config.rerefOverwrite = logical(handles.overwrite.Value);
mainHandles.config.rerefSearch = search;
if logical(handles.peak1Radio.Value)
    mainHandles.config.rerefPeakMethod = 1;
else
    mainHandles.config.rerefPeakMethod = 2;
end
mainHandles.config.rerefKernel = kernel;
mainHandles.config.rerefTorsion = logical(handles.fixTorsion.Value);
mainHandles.config.rerefTiltLow = tiltLow;
mainHandles.config.rerefTiltUp = tiltUp;
mainHandles.config.rerefTiltStep = tiltStep;

% Update handles structure
guidata(figureHandle, mainHandles);

close;

% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;

function kernel_Callback(hObject, eventdata, handles)
% hObject    handle to kernel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kernel as text
%        str2double(get(hObject,'String')) returns contents of kernel as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsOddNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be an odd, natural number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);


function tiltLow_Callback(hObject, eventdata, handles)
% hObject    handle to tiltLow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tiltLow as text
%        str2double(get(hObject,'String')) returns contents of tiltLow as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

function tiltUp_Callback(hObject, eventdata, handles)
% hObject    handle to tiltUp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tiltUp as text
%        str2double(get(hObject,'String')) returns contents of tiltUp as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in fixTorsion.
function fixTorsion_Callback(hObject, eventdata, handles)
% hObject    handle to fixTorsion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of fixTorsion
if logical(hObject.Value)
    handles.tiltLow.Enable = 'on';
    handles.tiltUp.Enable = 'on';
    handles.tiltStep.Enable = 'on';
else
    handles.tiltLow.Enable = 'off';
    handles.tiltUp.Enable = 'off';
    handles.tiltStep.Enable = 'off';
end


% --- Executes on button press in peak2Radio.
function peak2Radio_Callback(hObject, eventdata, handles)
% hObject    handle to peak2Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of peak2Radio
if logical(hObject.Value)
    handles.kernel.Enable = 'on';
else
    handles.kernel.Enable = 'off';
end


% --- Executes on button press in peak1Radio.
function peak1Radio_Callback(hObject, eventdata, handles)
% hObject    handle to peak1Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of peak1Radio
if logical(hObject.Value)
    handles.kernel.Enable = 'off';
else
    handles.kernel.Enable = 'on';
end

% --- Executes on button press in select.
function select_Callback(hObject, eventdata, handles)
% hObject    handle to select (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName, pathName, ~] = uigetfile('*.*', 'Upload Global Reference Frame', '');
if fileName == 0
    % User canceled.
    return;
end
handles.globalFullPath = fullfile(pathName, fileName);
handles.globalPath.String = [' ' fileName];

figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

if ~IsImageFile(handles.globalFullPath) && ~strcmp(fileName(end-3:end), '.mat')
    handles.globalPath.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    handles.globalPath.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a mat or image file.';
elseif size(fileName, 2) > 3 && strcmp(fileName(end-3:end), '.mat') && ...
        (~ismember('globalRef', who('-file', handles.globalFullPath)) && ...
         ~ismember('refFrame', who('-file', handles.globalFullPath)) && ...
         ~ismember('coarseRefFrame', who('-file', handles.globalFullPath)))
    handles.globalPath.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    handles.globalPath.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Mat file must contain variable called globalRef.';
else
    handles.globalPath.BackgroundColor = mainHandles.revasColors.background;
    handles.globalPath.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

function globalPath_Callback(hObject, eventdata, handles)
% hObject    handle to globalPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of globalPath as text
%        str2double(get(hObject,'String')) returns contents of globalPath as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = handles.globalFullPath;

if ~isempty(value) && ~IsImageFile(value) && ~strcmp(value, '.mat')
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
elseif size(value, 2) > 3 && strcmp(value(end-3:end), '.mat') && ismember('globalRef', who('-file', value))
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function globalPath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to globalPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
