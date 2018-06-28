function varargout = FilteringParameters(varargin)
% FILTERINGPARAMETERS MATLAB code for FilteringParameters.fig
%      FILTERINGPARAMETERS, by itself, creates a new FILTERINGPARAMETERS or raises the existing
%      singleton*.
%
%      H = FILTERINGPARAMETERS returns the handle to a new FILTERINGPARAMETERS or the handle to
%      the existing singleton*.
%
%      FILTERINGPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FILTERINGPARAMETERS.M with the given input arguments.
%
%      FILTERINGPARAMETERS('Property','Value',...) creates a new FILTERINGPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FilteringParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FilteringParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FilteringParameters

% Last Modified by GUIDE v2.5 31-May-2018 15:44:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FilteringParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @FilteringParameters_OutputFcn, ...
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

% --- Executes just before FilteringParameters is made visible.
function FilteringParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to filtparameters (see VARARGIN)

% Choose default command line output for filtparameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.overwrite.Value = mainHandles.config.filtOverwrite;
handles.verbosity.Value = mainHandles.config.filtVerbosity;
handles.maxGapDur.String = mainHandles.config.filtMaxGapDur;
handles.FirstPrefilter.Value = mainHandles.config.filtFirstPrefilter;
handles.SecondPrefilter.Value = mainHandles.config.filtSecondPrefilter;
handles.median1Radio.Value = mainHandles.config.filtEnableMedian1;
handles.sgo1Radio.Value = mainHandles.config.filtEnableSgo1;
handles.median1.String = mainHandles.config.filtMedian1;
handles.poly1.String = mainHandles.config.filtPoly1;
handles.kernel1.String = mainHandles.config.filtKernel1;
handles.median2Radio.Value = mainHandles.config.filtEnableMedian2;
handles.sgo2Radio.Value = mainHandles.config.filtEnableSgo2;
handles.noFilt2Radio.Value = mainHandles.config.filtEnableNoFilt2;
handles.median2.String = mainHandles.config.filtMedian2;
handles.poly2.String = mainHandles.config.filtPoly2;
handles.kernel2.String = mainHandles.config.filtKernel2;

% Set colors
% Main Background
handles.filtParameters.Color = mainHandles.colors{4,2};
handles.maxGapDur.BackgroundColor = mainHandles.colors{4,2};
handles.median1.BackgroundColor = mainHandles.colors{4,2};
handles.poly1.BackgroundColor = mainHandles.colors{4,2};
handles.kernel1.BackgroundColor = mainHandles.colors{4,2};
handles.median2.BackgroundColor = mainHandles.colors{4,2};
handles.poly2.BackgroundColor = mainHandles.colors{4,2};
handles.kernel2.BackgroundColor = mainHandles.colors{4,2};
% Box backgrounds
handles.titleBox.BackgroundColor = mainHandles.colors{4,3};
handles.usageBox.BackgroundColor = mainHandles.colors{4,3};
handles.filtBox.BackgroundColor = mainHandles.colors{4,3};
handles.filt0Box.BackgroundColor = mainHandles.colors{4,3};
handles.filt1Box.BackgroundColor = mainHandles.colors{4,3};
handles.filt1GroupBox.BackgroundColor = mainHandles.colors{4,3};
handles.filt2Box.BackgroundColor = mainHandles.colors{4,3};
handles.filt2GroupBox.BackgroundColor = mainHandles.colors{4,3};
handles.overwrite.BackgroundColor = mainHandles.colors{4,3};
handles.FirstPrefilter.BackgroundColor = mainHandles.colors{4,3};
handles.SecondPrefilter.BackgroundColor = mainHandles.colors{4,3};
handles.verbosity.BackgroundColor = mainHandles.colors{4,3};
handles.maxGapDurText.BackgroundColor = mainHandles.colors{4,3};
handles.maxGapDurTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.median1Radio.BackgroundColor = mainHandles.colors{4,3};
handles.median1Text.BackgroundColor = mainHandles.colors{4,3};
handles.median1TextSub.BackgroundColor = mainHandles.colors{4,3};
handles.sgo1Radio.BackgroundColor = mainHandles.colors{4,3};
handles.poly1Text.BackgroundColor = mainHandles.colors{4,3};
handles.kernel1Text.BackgroundColor = mainHandles.colors{4,3};
handles.kernel1TextSub.BackgroundColor = mainHandles.colors{4,3};
handles.median2Radio.BackgroundColor = mainHandles.colors{4,3};
handles.median2Text.BackgroundColor = mainHandles.colors{4,3};
handles.median2TextSub.BackgroundColor = mainHandles.colors{4,3};
handles.sgo2Radio.BackgroundColor = mainHandles.colors{4,3};
handles.poly2Text.BackgroundColor = mainHandles.colors{4,3};
handles.kernel2Text.BackgroundColor = mainHandles.colors{4,3};
handles.kernel2TextSub.BackgroundColor = mainHandles.colors{4,3};
handles.noFilt2Radio.BackgroundColor = mainHandles.colors{4,3};
% Box text
handles.titleBox.ForegroundColor = mainHandles.colors{4,5};
handles.usageBox.ForegroundColor = mainHandles.colors{4,5};
handles.filtBox.ForegroundColor = mainHandles.colors{4,5};
handles.filt0Box.ForegroundColor = mainHandles.colors{4,5};
handles.filt1Box.ForegroundColor = mainHandles.colors{4,5};
handles.filt1GroupBox.ForegroundColor = mainHandles.colors{4,5};
handles.filt2Box.ForegroundColor = mainHandles.colors{4,5};
handles.filt2GroupBox.ForegroundColor = mainHandles.colors{4,5};
handles.overwrite.ForegroundColor = mainHandles.colors{4,5};
handles.FirstPrefilter.ForegroundColor = mainHandles.colors{4,5};
handles.SecondPrefilter.ForegroundColor = mainHandles.colors{4,5};
handles.verbosity.ForegroundColor = mainHandles.colors{4,5};
handles.maxGapDurText.ForegroundColor = mainHandles.colors{4,5};
handles.maxGapDurTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.median1Radio.ForegroundColor = mainHandles.colors{4,5};
handles.median1Text.ForegroundColor = mainHandles.colors{4,5};
handles.median1TextSub.ForegroundColor = mainHandles.colors{4,5};
handles.sgo1Radio.ForegroundColor = mainHandles.colors{4,5};
handles.poly1Text.ForegroundColor = mainHandles.colors{4,5};
handles.kernel1Text.ForegroundColor = mainHandles.colors{4,5};
handles.kernel1TextSub.ForegroundColor = mainHandles.colors{4,5};
handles.median2Radio.ForegroundColor = mainHandles.colors{4,5};
handles.median2Text.ForegroundColor = mainHandles.colors{4,5};
handles.median2TextSub.ForegroundColor = mainHandles.colors{4,5};
handles.sgo2Radio.ForegroundColor = mainHandles.colors{4,5};
handles.poly2Text.ForegroundColor = mainHandles.colors{4,5};
handles.kernel2Text.ForegroundColor = mainHandles.colors{4,5};
handles.kernel2TextSub.ForegroundColor = mainHandles.colors{4,5};
handles.noFilt2Radio.ForegroundColor = mainHandles.colors{4,5};
% Save button
handles.save.BackgroundColor = mainHandles.colors{3,4};
handles.save.ForegroundColor = mainHandles.colors{3,2};
% Cancel button
handles.cancel.BackgroundColor = mainHandles.colors{2,4};
handles.cancel.ForegroundColor = mainHandles.colors{2,2};

% Update handles structure
guidata(hObject, handles);

% Check parameter validity and change colors if needed
maxGapDur_Callback(handles.maxGapDur, eventdata, handles);
median1_Callback(handles.median1, eventdata, handles);
poly1_Callback(handles.poly1, eventdata, handles);
kernel1_Callback(handles.kernel1, eventdata, handles);
median2_Callback(handles.median2, eventdata, handles);
poly2_Callback(handles.poly2, eventdata, handles);
kernel2_Callback(handles.kernel2, eventdata, handles);
median1Radio_Callback(handles.median1Radio, eventdata, handles);
median2Radio_Callback(handles.median2Radio, eventdata, handles);

% UIWAIT makes FilteringParameters wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = FilteringParameters_OutputFcn(hObject, eventdata, handles) 
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
% maxGapDur
maxGapDur = str2double(handles.maxGapDur.String);
if ~IsNaturalNumber(maxGapDur)
    errordlg('Maximum Gap Duration must be a natural number.', 'Invalid Parameter');
    return;
end

if logical(handles.median1Radio.Value)
    % median1
    median1 = str2double(handles.median1.String);
    if ~IsNaturalNumber(median1)
        errordlg('Filter #1 Median must be a natural number.', 'Invalid Parameter');
        return;
    end
elseif logical(handles.sgo2Radio.Value)
    % poly1
    poly1 = str2double(handles.poly1.String);
    if ~IsOddNaturalNumber(poly1)
        errordlg('Filter #1 Polynomial Order must be an odd natural number.', 'Invalid Parameter');
        return;
    end

    % kernel1
    kernel1 = str2double(handles.kernel1.String);
    if ~IsOddNaturalNumber(kernel1)
        errordlg('Filter #1 Kernel Size must be an odd natural number.', 'Invalid Parameter');
        return;
    end
end

if logical(handles.median2Radio.Value)
    % median2
    median2 = str2double(handles.median2.String);
    if ~IsNaturalNumber(median2)
        errordlg('Filter #2 Median must be a natural number.', 'Invalid Parameter');
        return;
    end
elseif logical(handles.sgo2Radio.Value)
    % poly2
    poly2 = str2double(handles.poly2.String);
    if ~IsOddNaturalNumber(poly2)
        errordlg('Filter #2 Polynomial Order must be an odd natural number.', 'Invalid Parameter');
        return;
    end

    % kernel2
    kernel2 = str2double(handles.kernel2.String);
    if ~IsOddNaturalNumber(kernel2)
        errordlg('Filter #2 Kernel Size must be an odd natural number.', 'Invalid Parameter');
        return;
    end
end

% Save new configurations
mainHandles.config.filtMaxGapDur = str2double(handles.maxGapDur.String);
mainHandles.config.filtOverwrite = logical(handles.overwrite.Value);
mainHandles.config.filtVerbosity = logical(handles.verbosity.Value);
mainHandles.config.filtFirstPrefilter = logical(handles.FirstPrefilter.Value);
mainHandles.config.filtSecondPrefilter = logical(handles.SecondPrefilter.Value);
mainHandles.config.filtEnableMedian1 = logical(handles.median1Radio.Value);
mainHandles.config.filtEnableSgo1 = logical(handles.sgo1Radio.Value);
mainHandles.config.filtMedian1 = str2double(handles.median1.String);
mainHandles.config.filtPoly1 = str2double(handles.poly1.String);
mainHandles.config.filtKernel1 = str2double(handles.kernel1.String);
mainHandles.config.filtEnableMedian2 = logical(handles.median2Radio.Value);
mainHandles.config.filtEnableSgo2 = logical(handles.sgo2Radio.Value);
mainHandles.config.filtEnableNoFilt2 = logical(handles.noFilt2Radio.Value);
mainHandles.config.filtMedian2 = str2double(handles.median2.String);
mainHandles.config.filtPoly2 = str2double(handles.poly2.String);
mainHandles.config.filtKernel2 = str2double(handles.kernel2.String);

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

function maxGapDur_Callback(hObject, eventdata, handles)
% hObject    handle to maxGapDur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxGapDur as text
%        str2double(get(hObject,'String')) returns contents of maxGapDur as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function maxGapDur_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxGapDur (see GCBO)
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
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

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

function median1_Callback(hObject, eventdata, handles)
% hObject    handle to median1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of median1 as text
%        str2double(get(hObject,'String')) returns contents of median1 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function median1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to median1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function poly1_Callback(hObject, eventdata, handles)
% hObject    handle to poly1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of poly1 as text
%        str2double(get(hObject,'String')) returns contents of poly1 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsOddNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be an odd natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function poly1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to poly1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function kernel1_Callback(hObject, eventdata, handles)
% hObject    handle to kernel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kernel1 as text
%        str2double(get(hObject,'String')) returns contents of kernel1 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsOddNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be an odd natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function kernel1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kernel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function median2_Callback(hObject, eventdata, handles)
% hObject    handle to median2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of median2 as text
%        str2double(get(hObject,'String')) returns contents of median2 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be a natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function median2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to median2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function poly2_Callback(hObject, eventdata, handles)
% hObject    handle to poly2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of poly2 as text
%        str2double(get(hObject,'String')) returns contents of poly2 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsOddNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be an odd natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function poly2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to poly2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function kernel2_Callback(hObject, eventdata, handles)
% hObject    handle to kernel2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kernel2 as text
%        str2double(get(hObject,'String')) returns contents of kernel2 as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsOddNaturalNumber(value)
    hObject.BackgroundColor = mainHandles.colors{2,4};
    hObject.ForegroundColor = mainHandles.colors{2,2};
    hObject.TooltipString = 'Must be an odd natural number.';
else
    hObject.BackgroundColor = mainHandles.colors{4,2};
    hObject.ForegroundColor = mainHandles.colors{4,5};
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function kernel2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kernel2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in median1Radio.
function median1Radio_Callback(hObject, eventdata, handles)
% hObject    handle to median1Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of median1Radio
if logical(hObject.Value)
    handles.median1.Enable = 'on';
    handles.poly1.Enable = 'off';
    handles.kernel1.Enable = 'off';
else
    handles.median1.Enable = 'off';
    handles.poly1.Enable = 'on';
    handles.kernel1.Enable = 'on';
end


% --- Executes on button press in sgo1Radio.
function sgo1Radio_Callback(hObject, eventdata, handles)
% hObject    handle to sgo1Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of sgo1Radio
if ~logical(hObject.Value)
    handles.median1.Enable = 'on';
    handles.poly1.Enable = 'off';
    handles.kernel1.Enable = 'off';
else
    handles.median1.Enable = 'off';
    handles.poly1.Enable = 'on';
    handles.kernel1.Enable = 'on';
end

% --- Executes on button press in median2Radio.
function median2Radio_Callback(hObject, eventdata, handles)
% hObject    handle to median2Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of median2Radio
if logical(hObject.Value)
    handles.median2.Enable = 'on';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
elseif logical(handles.sgo2Radio.Value)
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'on';
    handles.kernel2.Enable = 'on';
else
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
end

% --- Executes on button press in sgo2Radio.
function sgo2Radio_Callback(hObject, eventdata, handles)
% hObject    handle to sgo2Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of sgo2Radio
if logical(handles.median2Radio.Value)
    handles.median2.Enable = 'on';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
elseif logical(hObject.Value)
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'on';
    handles.kernel2.Enable = 'on';
else
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
end


% --- Executes on button press in noFilt2Radio.
function noFilt2Radio_Callback(hObject, eventdata, handles)
% hObject    handle to noFilt2Radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of noFilt2Radio
if logical(handles.median2Radio.Value)
    handles.median2.Enable = 'on';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
elseif logical(handles.sgo2Radio.Value)
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'on';
    handles.kernel2.Enable = 'on';
else
    handles.median2.Enable = 'off';
    handles.poly2.Enable = 'off';
    handles.kernel2.Enable = 'off';
end


% --- Executes on button press in FirstPrefilter.
function FirstPrefilter_Callback(hObject, eventdata, handles)
% hObject    handle to FirstPrefilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of FirstPrefilter


% --- Executes on button press in SecondPrefilter.
function SecondPrefilter_Callback(hObject, eventdata, handles)
% hObject    handle to SecondPrefilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SecondPrefilter
