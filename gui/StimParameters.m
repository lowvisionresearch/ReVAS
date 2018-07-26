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

% Last Modified by GUIDE v2.5 14-Nov-2017 17:43:28

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
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.verbosity.Value = mainHandles.config.stimVerbosity;
handles.overwrite.Value = mainHandles.config.stimOverwrite;
handles.upload.Value = mainHandles.config.stimOption1;
handles.cross.Value = mainHandles.config.stimOption2;
handles.stimPath.String = mainHandles.config.stimPath;
handles.size.String = mainHandles.config.stimSize;
handles.thick.String = mainHandles.config.stimThick;
handles.length.String = mainHandles.config.stimRectangleX;
handles.width.String = mainHandles.config.stimRectangleY;
handles.rectangle.Value = mainHandles.config.stimUseRectangle;
handles.stimFullPath = mainHandles.config.stimFullPath;




% set a proper size for the main GUI window. a
handles.stimParameters.Units = 'normalized';
handles.stimParameters.OuterPosition = mainHandles.GUIposition.stimParameters;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.stimParameters);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.stimParameters.Color = revasColors.background;
handles.stimPath.BackgroundColor = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.usageBox.BackgroundColor = revasColors.boxBackground;
handles.stimBox.BackgroundColor = revasColors.boxBackground;
handles.stimGroup.BackgroundColor = revasColors.boxBackground;
handles.overwrite.BackgroundColor = revasColors.boxBackground;
handles.verbosity.BackgroundColor = revasColors.boxBackground;
handles.upload.BackgroundColor = revasColors.boxBackground;
handles.cross.BackgroundColor = revasColors.boxBackground;
handles.sizeText.BackgroundColor = revasColors.boxBackground;
handles.thickText.BackgroundColor = revasColors.boxBackground;
handles.lengthText.BackgroundColor = revasColors.boxBackground;
handles.widthText.BackgroundColor = revasColors.boxBackground;
handles.rectangle.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.usageBox.ForegroundColor = revasColors.text;
handles.stimBox.ForegroundColor = revasColors.text;
handles.overwrite.ForegroundColor = revasColors.text;
handles.verbosity.ForegroundColor = revasColors.text;
handles.upload.ForegroundColor = revasColors.text;
handles.cross.ForegroundColor = revasColors.text;
handles.sizeText.ForegroundColor = revasColors.text;
handles.thickText.ForegroundColor = revasColors.text;
handles.lengthText.ForegroundColor = revasColors.text;
handles.widthText.ForegroundColor = revasColors.text;
handles.rectangle.ForegroundColor = revasColors.text;
handles.stimPath.ForegroundColor = revasColors.text;
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
stimPath_Callback(handles.stimPath, eventdata, handles);
size_Callback(handles.size, eventdata, handles);
thick_Callback(handles.thick, eventdata, handles);
length_Callback(handles.length, eventdata, handles);
width_Callback(handles.width, eventdata, handles);
upload_Callback(handles.upload, eventdata, handles);
rectangle_Callback(handles.rectangle, eventdata, handles);

% UIWAIT makes StimParameters wait for user response (see UIRESUME)
% uiwait(handles.stimParameters);


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
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

% Validate new configurations
size = str2double(handles.size.String);
thick = str2double(handles.thick.String);
length = str2double(handles.length.String);
width = str2double(handles.width.String);

if logical(handles.upload.Value)
    % full path
    if ~IsImageFile(handles.stimFullPath)
        errordlg('Uploaded Stimulus Image must be an image file.', 'Invalid Parameter');
        return;
    end
else
    % size
    if ~IsOddNaturalNumber(size)
        errordlg('Stimulus Size must be an odd, natural number.', 'Invalid Parameter');
        return;
    end

    % thick
    if ~IsOddNaturalNumber(thick)
        errordlg('Cross Thickness must be an odd, natural number.', 'Invalid Parameter');
        return;
    end    
end

if logical(handles.rectangle.Value)
    % length
    if ~IsOddNaturalNumber(length)
        errordlg('Length must be an odd, natural number.', 'Invalid Parameter');
        return;
    end
    
    % width
    if ~IsOddNaturalNumber(width)
        errordlg('Width must be an odd, natural number.', 'Invalid Parameter');
        return;
    end
end

% Save new configurations
mainHandles.config.stimVerbosity = logical(handles.verbosity.Value);
mainHandles.config.stimOverwrite = logical(handles.overwrite.Value);
mainHandles.config.stimOption1 = logical(handles.upload.Value);
mainHandles.config.stimOption2 = logical(handles.cross.Value);
mainHandles.config.stimPath = handles.stimPath.String;
mainHandles.config.stimFullPath = handles.stimFullPath;
mainHandles.config.stimSize = size;
mainHandles.config.stimThick = thick;
mainHandles.config.stimRectangleX = length;
mainHandles.config.stimRectangleY = width;
mainHandles.config.stimUseRectangle = logical(handles.rectangle.Value);

% Update handles structure
guidata(figureHandle, mainHandles);

close;

% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;


function size_Callback(hObject, eventdata, handles)
% hObject    handle to size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of size as text
%        str2double(get(hObject,'String')) returns contents of size as a double
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

% --- Executes during object creation, after setting all properties.
function size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function thick_Callback(hObject, eventdata, handles)
% hObject    handle to thick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of thick as text
%        str2double(get(hObject,'String')) returns contents of thick as a double
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

% --- Executes during object creation, after setting all properties.
function thick_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in select.
function select_Callback(hObject, eventdata, handles)
% hObject    handle to select (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName, pathName, ~] = uigetfile('*.*', 'Upload Stimulus Image', '');
if fileName == 0
    % User canceled.
    return;
end
handles.stimFullPath = fullfile(pathName, fileName);
handles.stimPath.String = [' ' fileName];

figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

if ~IsImageFile(fullfile(pathName, fileName))
    handles.stimPath.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    handles.stimPath.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be an image file.';
else
    handles.stimPath.BackgroundColor = mainHandles.revasColors.background;
    handles.stimPath.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);


function stimPath_Callback(hObject, eventdata, handles)
% hObject    handle to stimPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stimPath as text
%        str2double(get(hObject,'String')) returns contents of stimPath as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = handles.stimFullPath;

if ~isempty(value) && ~IsImageFile(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function stimPath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stimPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in upload.
function upload_Callback(hObject, eventdata, handles)
% hObject    handle to upload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of upload
if logical(hObject.Value)
    handles.stimPath.Enable = 'inactive';
    handles.select.Enable = 'on';
    handles.size.Enable = 'off';
    handles.thick.Enable = 'off';
else
    handles.stimPath.Enable = 'off';
    handles.select.Enable = 'off';
    handles.size.Enable = 'on';
    handles.thick.Enable = 'on';
end


% --- Executes on button press in cross.
function cross_Callback(hObject, eventdata, handles)
% hObject    handle to cross (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cross
if ~logical(hObject.Value)
    handles.stimPath.Enable = 'inactive';
    handles.select.Enable = 'on';
    handles.size.Enable = 'off';
    handles.thick.Enable = 'off';
else
    handles.stimPath.Enable = 'off';
    handles.select.Enable = 'off';
    handles.size.Enable = 'on';
    handles.thick.Enable = 'on';
end


function width_Callback(hObject, eventdata, handles)
% hObject    handle to width (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of width as text
%        str2double(get(hObject,'String')) returns contents of width as a double
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


% --- Executes during object creation, after setting all properties.
function width_CreateFcn(hObject, eventdata, handles)
% hObject    handle to width (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function length_Callback(hObject, eventdata, handles)
% hObject    handle to length (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of length as text
%        str2double(get(hObject,'String')) returns contents of length as a double
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


% --- Executes during object creation, after setting all properties.
function length_CreateFcn(hObject, eventdata, handles)
% hObject    handle to length (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rectangle.
function rectangle_Callback(hObject, eventdata, handles)
% hObject    handle to rectangle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rectangle
if logical(hObject.Value)
    handles.length.Enable = 'on';
    handles.width.Enable = 'on';
else
    handles.length.Enable = 'off';
    handles.width.Enable = 'off';
end


% --- Executes on button press in save.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in cancel.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
