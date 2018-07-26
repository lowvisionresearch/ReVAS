function varargout = FineParameters(varargin)
% FINEPARAMETERS MATLAB code for FineParameters.fig
%      FINEPARAMETERS, by itself, creates a new FINEPARAMETERS or raises the existing
%      singleton*.
%
%      H = FINEPARAMETERS returns the handle to a new FINEPARAMETERS or the handle to
%      the existing singleton*.
%
%      FINEPARAMETERS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FINEPARAMETERS.M with the given input arguments.
%
%      FINEPARAMETERS('Property','Value',...) creates a new FINEPARAMETERS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FineParameters_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FineParameters_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FineParameters

% Last Modified by GUIDE v2.5 05-Aug-2017 16:47:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FineParameters_OpeningFcn, ...
                   'gui_OutputFcn',  @FineParameters_OutputFcn, ...
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


% --- Executes just before FineParameters is made visible.
function FineParameters_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to FineParameters (see VARARGIN)

% Choose default command line output for FineParameters
handles.output = hObject;

% Loading previously saved or default parameters
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);

handles.overwrite.Value = mainHandles.config.fineOverwrite;
handles.verbosity.Value = mainHandles.config.fineVerbosity;
handles.numIterations.String = mainHandles.config.fineNumIterations;
handles.stripHeight.String = mainHandles.config.fineStripHeight;
handles.stripWidth.String = mainHandles.config.fineStripWidth;
handles.samplingRate.String = mainHandles.config.fineSamplingRate;
handles.maxPeakRatio.String = mainHandles.config.fineMaxPeakRatio;
handles.minPeakThreshold.String = mainHandles.config.fineMinPeakThreshold;
handles.adaptiveSearch.Value = mainHandles.config.fineAdaptiveSearch;
handles.scalingFactor.String = mainHandles.config.fineScalingFactor;
handles.searchWindowHeight.String = mainHandles.config.fineSearchWindowHeight;
handles.subpixelInterp.Value = mainHandles.config.fineSubpixelInterp;
handles.neighborhoodSize.String = mainHandles.config.fineNeighborhoodSize;
handles.subpixelDepth.String = mainHandles.config.fineSubpixelDepth;


% set a proper size for the main GUI window. a
handles.fineParameters.Units = 'normalized';
handles.fineParameters.OuterPosition = mainHandles.GUIposition.fineParameters;

% set font size and size and position of the GUI
InitGUIHelper(mainHandles, handles.fineParameters);


% Set colors
revasColors = mainHandles.revasColors;
% Main Background
handles.fineParameters.Color = revasColors.background;
handles.numIterations.BackgroundColor = revasColors.background;
handles.stripHeight.BackgroundColor = revasColors.background;
handles.stripWidth.BackgroundColor = revasColors.background;
handles.samplingRate.BackgroundColor = revasColors.background;
handles.maxPeakRatio.BackgroundColor = revasColors.background;
handles.minPeakThreshold.BackgroundColor = revasColors.background;
handles.scalingFactor.BackgroundColor = revasColors.background;
handles.searchWindowHeight.BackgroundColor = revasColors.background;
handles.neighborhoodSize.BackgroundColor = revasColors.background;
handles.subpixelDepth.BackgroundColor = revasColors.background;
% Box backgrounds
handles.titleBox.BackgroundColor = revasColors.boxBackground;
handles.usageBox.BackgroundColor = revasColors.boxBackground;
handles.fineBox.BackgroundColor = revasColors.boxBackground;
handles.stripBox.BackgroundColor = revasColors.boxBackground;
handles.peakBox.BackgroundColor = revasColors.boxBackground;
handles.adaptiveBox.BackgroundColor = revasColors.boxBackground;
handles.interpBox.BackgroundColor = revasColors.boxBackground;
handles.overwrite.BackgroundColor = revasColors.boxBackground;
handles.verbosity.BackgroundColor = revasColors.boxBackground;
handles.iterText.BackgroundColor = revasColors.boxBackground;
handles.heightText.BackgroundColor = revasColors.boxBackground;
handles.heightTextSub.BackgroundColor = revasColors.boxBackground;
handles.widthText.BackgroundColor = revasColors.boxBackground;
handles.widthTextSub.BackgroundColor = revasColors.boxBackground;
handles.sampleText.BackgroundColor = revasColors.boxBackground;
handles.sampleTextSub.BackgroundColor = revasColors.boxBackground;
handles.ratioText.BackgroundColor = revasColors.boxBackground;
handles.threshText.BackgroundColor = revasColors.boxBackground;
handles.adaptiveSearch.BackgroundColor = revasColors.boxBackground;
handles.scaleText.BackgroundColor = revasColors.boxBackground;
handles.searchWindowHeightText.BackgroundColor = revasColors.boxBackground;
handles.searchWindowHeightTextSub.BackgroundColor = revasColors.boxBackground;
handles.subpixelInterp.BackgroundColor = revasColors.boxBackground;
handles.neighborText.BackgroundColor = revasColors.boxBackground;
handles.neighborTextSub.BackgroundColor = revasColors.boxBackground;
handles.depthText.BackgroundColor = revasColors.boxBackground;
% Box text
handles.titleBox.ForegroundColor = revasColors.text;
handles.usageBox.ForegroundColor = revasColors.text;
handles.fineBox.ForegroundColor = revasColors.text;
handles.stripBox.ForegroundColor = revasColors.text;
handles.peakBox.ForegroundColor = revasColors.text;
handles.adaptiveBox.ForegroundColor = revasColors.text;
handles.interpBox.ForegroundColor = revasColors.text;
handles.overwrite.ForegroundColor = revasColors.text;
handles.verbosity.ForegroundColor = revasColors.text;
handles.iterText.ForegroundColor = revasColors.text;
handles.heightText.ForegroundColor = revasColors.text;
handles.heightTextSub.ForegroundColor = revasColors.text;
handles.widthText.ForegroundColor = revasColors.text;
handles.widthTextSub.ForegroundColor = revasColors.text;
handles.sampleText.ForegroundColor = revasColors.text;
handles.sampleTextSub.ForegroundColor = revasColors.text;
handles.ratioText.ForegroundColor = revasColors.text;
handles.threshText.ForegroundColor = revasColors.text;
handles.adaptiveSearch.ForegroundColor = revasColors.text;
handles.scaleText.ForegroundColor = revasColors.text;
handles.searchWindowHeightText.ForegroundColor = revasColors.text;
handles.searchWindowHeightTextSub.ForegroundColor = revasColors.text;
handles.subpixelInterp.ForegroundColor = revasColors.text;
handles.neighborText.ForegroundColor = revasColors.text;
handles.neighborTextSub.ForegroundColor = revasColors.text;
handles.depthText.ForegroundColor = revasColors.text;
handles.numIterations.ForegroundColor = revasColors.text;
handles.stripHeight.ForegroundColor = revasColors.text;
handles.stripWidth.ForegroundColor = revasColors.text;
handles.samplingRate.ForegroundColor = revasColors.text;
handles.maxPeakRatio.ForegroundColor = revasColors.text;
handles.minPeakThreshold.ForegroundColor = revasColors.text;
handles.scalingFactor.ForegroundColor = revasColors.text;
handles.searchWindowHeight.ForegroundColor = revasColors.text;
handles.neighborhoodSize.ForegroundColor = revasColors.text;
handles.subpixelDepth.ForegroundColor = revasColors.text;
% Save button
handles.save.BackgroundColor = revasColors.pushButtonBackground;
handles.save.ForegroundColor = revasColors.pushButtonText;
% Cancel button
handles.cancel.BackgroundColor = revasColors.pushButtonBackground;
handles.cancel.ForegroundColor = revasColors.pushButtonText;

% Update handles structure
guidata(hObject, handles);

% Check parameter validity and change colors if needed
scalingFactor_Callback(handles.scalingFactor, eventdata, handles);
searchWindowHeight_Callback(handles.searchWindowHeight, eventdata, handles);
neighborhoodSize_Callback(handles.neighborhoodSize, eventdata, handles);
subpixelDepth_Callback(handles.subpixelDepth, eventdata, handles);
adaptiveSearch_Callback(handles.adaptiveSearch, eventdata, handles);
subpixelInterp_Callback(handles.subpixelInterp, eventdata, handles);

% UIWAIT makes FineParameters wait for user response (see UIRESUME)
% uiwait(handles.fineParameters);

% --- Outputs from this function are returned to the command line.
function varargout = FineParameters_OutputFcn(hObject, eventdata, handles) 
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
% numIterations
numIterations = str2double(handles.numIterations.String);
if ~IsNaturalNumber(numIterations)
    errordlg('Number of Iterations must be a natural number.', 'Invalid Parameter');
    return;
end

% stripHeight
stripHeight = str2double(handles.stripHeight.String);
if ~IsNaturalNumber(stripHeight)
    errordlg('Strip Height must be a natural number.', 'Invalid Parameter');
    return;
end

% stripWidth
stripWidth = str2double(handles.stripWidth.String);
if ~IsNaturalNumber(stripHeight)
    errordlg('Strip Width must be a natural number.', 'Invalid Parameter');
    return;
end

% samplingRate
samplingRate = str2double(handles.samplingRate.String);
if ~IsPositiveRealNumber(samplingRate)
    errordlg('Sampling Rate must be a positive, real number.', 'Invalid Parameter');
    return;
end

% MaxPeakRatio
maxPeakRatio = str2double(handles.maxPeakRatio.String);
if ~IsPositiveRealNumber(maxPeakRatio)
    errordlg('Minimum Peak Ratio must be a positive, real number.', 'Invalid Parameter');
    return;
end

% minPeakThreshold
minPeakThreshold = str2double(handles.minPeakThreshold.String);
if ~IsPositiveRealNumber(minPeakThreshold)
    errordlg('Minimum Peak Threshold must be a positive, real number.', 'Invalid Parameter');
    return;
end

if logical(handles.adaptiveSearch.Value)
    % scalingFactor
    scalingFactor = str2double(handles.scalingFactor.String);
    if ~IsPositiveRealNumber(scalingFactor)
        errordlg('Scaling Factor must be a positive, real number.', 'Invalid Parameter');
        return;
    end

    % searchWindowHeight
    searchWindowHeight = str2double(handles.searchWindowHeight.String);
    if ~IsNaturalNumber(searchWindowHeight)
        errordlg('Search Window Height must be a natural number.', 'Invalid Parameter');
        return;
    end
end

if logical(handles.subpixelInterp.Value)
    % neighborhoodSize
    neighborhoodSize = str2double(handles.neighborhoodSize.String);
    if ~IsNaturalNumber(neighborhoodSize)
        errordlg('Neighborhood Size must be a natural number.', 'Invalid Parameter');
        return;
    end

    % subpixelDepth
    subpixelDepth = str2double(handles.subpixelDepth.String);
    if ~IsPositiveRealNumber(subpixelDepth)
        errordlg('Subpixel Depth must be a positive, real number.', 'Invalid Parameter');
        return;
    end
end

% Save new configurations
mainHandles.config.fineOverwrite = logical(handles.overwrite.Value);
mainHandles.config.fineVerbosity = logical(handles.verbosity.Value);
mainHandles.config.fineNumIterations = str2double(handles.numIterations.String);
mainHandles.config.fineStripHeight = str2double(handles.stripHeight.String);
mainHandles.config.fineStripWidth = str2double(handles.stripWidth.String);
mainHandles.config.fineSamplingRate = str2double(handles.samplingRate.String);
mainHandles.config.fineMaxPeakRatio = str2double(handles.maxPeakRatio.String);
mainHandles.config.fineMinPeakThreshold = str2double(handles.minPeakThreshold.String);
mainHandles.config.fineAdaptiveSearch = logical(handles.adaptiveSearch.Value);
mainHandles.config.fineScalingFactor = str2double(handles.scalingFactor.String);
mainHandles.config.fineSearchWindowHeight = str2double(handles.searchWindowHeight.String);
mainHandles.config.fineSubpixelInterp = logical(handles.subpixelInterp.Value);
mainHandles.config.fineNeighborhoodSize = str2double(handles.neighborhoodSize.String);
mainHandles.config.fineSubpixelDepth = str2double(handles.subpixelDepth.String);

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


% --- Executes on button press in subpixelInterp.
function subpixelInterp_Callback(hObject, eventdata, handles)
% hObject    handle to subpixelInterp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of subpixelInterp
if get(hObject,'Value') == 1
    handles.neighborhoodSize.Enable = 'on';
    handles.subpixelDepth.Enable = 'on';
else
    handles.neighborhoodSize.Enable = 'off';
    handles.subpixelDepth.Enable = 'off';
end


% --- Executes on button press in verbosity.
function verbosity_Callback(hObject, eventdata, handles)
% hObject    handle to verbosity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of verbosity



function numIterations_Callback(hObject, eventdata, handles)
% hObject    handle to numIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numIterations as text
%        str2double(get(hObject,'String')) returns contents of numIterations as a double
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
function numIterations_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stripHeight_Callback(hObject, eventdata, handles)
% hObject    handle to stripHeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stripHeight as text
%        str2double(get(hObject,'String')) returns contents of stripHeight as a double
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
function stripHeight_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stripHeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stripWidth_Callback(hObject, eventdata, handles)
% hObject    handle to stripWidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stripWidth as text
%        str2double(get(hObject,'String')) returns contents of stripWidth as a double
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
function stripWidth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stripWidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function samplingRate_Callback(hObject, eventdata, handles)
% hObject    handle to samplingRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of samplingRate as text
%        str2double(get(hObject,'String')) returns contents of samplingRate as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function samplingRate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to samplingRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function maxPeakRatio_Callback(hObject, eventdata, handles)
% hObject    handle to MaxPeakRatio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MaxPeakRatio as text
%        str2double(get(hObject,'String')) returns contents of MaxPeakRatio as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function maxPeakRatio_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MaxPeakRatio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function minPeakThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to minPeakThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minPeakThreshold as text
%        str2double(get(hObject,'String')) returns contents of minPeakThreshold as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function minPeakThreshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minPeakThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in adaptiveSearch.
function adaptiveSearch_Callback(hObject, eventdata, handles)
% hObject    handle to adaptiveSearch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of adaptiveSearch
if get(hObject,'Value') == 1
    handles.scalingFactor.Enable = 'on';
    handles.searchWindowHeight.Enable = 'on';
else
    handles.scalingFactor.Enable = 'off';
    handles.searchWindowHeight.Enable = 'off';
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
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
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



function searchWindowHeight_Callback(hObject, eventdata, handles)
% hObject    handle to searchWindowHeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of searchWindowHeight as text
%        str2double(get(hObject,'String')) returns contents of searchWindowHeight as a double
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
function searchWindowHeight_CreateFcn(hObject, eventdata, handles)
% hObject    handle to searchWindowHeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function neighborhoodSize_Callback(hObject, eventdata, handles)
% hObject    handle to neighborhoodSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of neighborhoodSize as text
%        str2double(get(hObject,'String')) returns contents of neighborhoodSize as a double
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
function neighborhoodSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to neighborhoodSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function subpixelDepth_Callback(hObject, eventdata, handles)
% hObject    handle to subpixelDepth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of subpixelDepth as text
%        str2double(get(hObject,'String')) returns contents of subpixelDepth as a double
figureHandle = findobj(0, 'tag', 'revas');
mainHandles = guidata(figureHandle);
value = str2double(hObject.String);

if ~IsPositiveRealNumber(value)
    hObject.BackgroundColor = mainHandles.revasColors.abortButtonBackground;
    hObject.ForegroundColor = mainHandles.revasColors.abortButtonText;
    hObject.TooltipString = 'Must be a positive, real number.';
else
    hObject.BackgroundColor = mainHandles.revasColors.background;
    hObject.ForegroundColor = mainHandles.revasColors.text;
    hObject.TooltipString = '';
end

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function subpixelDepth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to subpixelDepth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
