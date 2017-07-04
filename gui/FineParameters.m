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

% Last Modified by GUIDE v2.5 01-Jul-2017 00:02:08

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
figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

handles.overwrite.Value = mainHandles.config.fineOverwrite;
handles.verbosity.Value = mainHandles.config.fineVerbosity;
handles.numIterations.String = mainHandles.config.fineNumIterations;
handles.stripHeight.String = mainHandles.config.fineStripHeight;
handles.stripWidth.String = mainHandles.config.fineStripWidth;
handles.samplingRate.String = mainHandles.config.fineSamplingRate;
handles.minPeakRatio.String = mainHandles.config.fineMinPeakRatio;
handles.minPeakThreshold.String = mainHandles.config.fineMinPeakThreshold;
handles.adaptiveSearch.Value = mainHandles.config.fineAdaptiveSearch;
handles.scalingFactor.String = mainHandles.config.fineScalingFactor;
handles.searchWindowHeight.String = mainHandles.config.fineSearchWindowHeight;
handles.subpixelInterp.Value = mainHandles.config.fineSubpixelInterp;
handles.neighborhoodSize.String = mainHandles.config.fineNeighborhoodSize;
handles.subpixelDepth.String = mainHandles.config.fineSubpixelDepth;

if logical(handles.adaptiveSearch.Value)
    handles.scalingFactor.Enable = 'on';
    handles.searchWindowHeight.Enable = 'on';
else
    handles.scalingFactor.Enable = 'off';
    handles.searchWindowHeight.Enable = 'off';
end

if logical(handles.subpixelInterp.Value)
    handles.neighborhoodSize.Enable = 'on';
    handles.subpixelDepth.Enable = 'on';
else
    handles.neighborhoodSize.Enable = 'off';
    handles.subpixelDepth.Enable = 'off';
end

% Set colors
% Main Background
handles.fineParameters.Color = mainHandles.colors{4,2};
handles.numIterations.BackgroundColor = mainHandles.colors{4,2};
handles.stripHeight.BackgroundColor = mainHandles.colors{4,2};
handles.stripWidth.BackgroundColor = mainHandles.colors{4,2};
handles.samplingRate.BackgroundColor = mainHandles.colors{4,2};
handles.minPeakRatio.BackgroundColor = mainHandles.colors{4,2};
handles.minPeakThreshold.BackgroundColor = mainHandles.colors{4,2};
handles.scalingFactor.BackgroundColor = mainHandles.colors{4,2};
handles.searchWindowHeight.BackgroundColor = mainHandles.colors{4,2};
handles.neighborhoodSize.BackgroundColor = mainHandles.colors{4,2};
handles.subpixelDepth.BackgroundColor = mainHandles.colors{4,2};
% Box backgrounds
handles.titleBox.BackgroundColor = mainHandles.colors{4,3};
handles.usageBox.BackgroundColor = mainHandles.colors{4,3};
handles.fineBox.BackgroundColor = mainHandles.colors{4,3};
handles.stripBox.BackgroundColor = mainHandles.colors{4,3};
handles.peakBox.BackgroundColor = mainHandles.colors{4,3};
handles.adaptiveBox.BackgroundColor = mainHandles.colors{4,3};
handles.interpBox.BackgroundColor = mainHandles.colors{4,3};
handles.overwrite.BackgroundColor = mainHandles.colors{4,3};
handles.verbosity.BackgroundColor = mainHandles.colors{4,3};
handles.iterText.BackgroundColor = mainHandles.colors{4,3};
handles.heightText.BackgroundColor = mainHandles.colors{4,3};
handles.heightTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.widthText.BackgroundColor = mainHandles.colors{4,3};
handles.widthTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.sampleText.BackgroundColor = mainHandles.colors{4,3};
handles.sampleTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.ratioText.BackgroundColor = mainHandles.colors{4,3};
handles.threshText.BackgroundColor = mainHandles.colors{4,3};
handles.adaptiveSearch.BackgroundColor = mainHandles.colors{4,3};
handles.scaleText.BackgroundColor = mainHandles.colors{4,3};
handles.searchWindowHeightText.BackgroundColor = mainHandles.colors{4,3};
handles.searchWindowHeightTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.subpixelInterp.BackgroundColor = mainHandles.colors{4,3};
handles.neighborText.BackgroundColor = mainHandles.colors{4,3};
handles.neighborTextSub.BackgroundColor = mainHandles.colors{4,3};
handles.depthText.BackgroundColor = mainHandles.colors{4,3};
% Box text
handles.titleBox.ForegroundColor = mainHandles.colors{4,5};
handles.usageBox.ForegroundColor = mainHandles.colors{4,5};
handles.fineBox.ForegroundColor = mainHandles.colors{4,5};
handles.stripBox.ForegroundColor = mainHandles.colors{4,5};
handles.peakBox.ForegroundColor = mainHandles.colors{4,5};
handles.adaptiveBox.ForegroundColor = mainHandles.colors{4,5};
handles.interpBox.ForegroundColor = mainHandles.colors{4,5};
handles.overwrite.ForegroundColor = mainHandles.colors{4,5};
handles.verbosity.ForegroundColor = mainHandles.colors{4,5};
handles.iterText.ForegroundColor = mainHandles.colors{4,5};
handles.heightText.ForegroundColor = mainHandles.colors{4,5};
handles.heightTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.widthText.ForegroundColor = mainHandles.colors{4,5};
handles.widthTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.sampleText.ForegroundColor = mainHandles.colors{4,5};
handles.sampleTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.ratioText.ForegroundColor = mainHandles.colors{4,5};
handles.threshText.ForegroundColor = mainHandles.colors{4,5};
handles.adaptiveSearch.ForegroundColor = mainHandles.colors{4,5};
handles.scaleText.ForegroundColor = mainHandles.colors{4,5};
handles.searchWindowHeightText.ForegroundColor = mainHandles.colors{4,5};
handles.searchWindowHeightTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.subpixelInterp.ForegroundColor = mainHandles.colors{4,5};
handles.neighborText.ForegroundColor = mainHandles.colors{4,5};
handles.neighborTextSub.ForegroundColor = mainHandles.colors{4,5};
handles.depthText.ForegroundColor = mainHandles.colors{4,5};
handles.numIterations.ForegroundColor = mainHandles.colors{4,5};
handles.stripHeight.ForegroundColor = mainHandles.colors{4,5};
handles.stripWidth.ForegroundColor = mainHandles.colors{4,5};
handles.samplingRate.ForegroundColor = mainHandles.colors{4,5};
handles.minPeakRatio.ForegroundColor = mainHandles.colors{4,5};
handles.minPeakThreshold.ForegroundColor = mainHandles.colors{4,5};
handles.scalingFactor.ForegroundColor = mainHandles.colors{4,5};
handles.searchWindowHeight.ForegroundColor = mainHandles.colors{4,5};
handles.neighborhoodSize.ForegroundColor = mainHandles.colors{4,5};
handles.subpixelDepth.ForegroundColor = mainHandles.colors{4,5};
% Save button
handles.save.BackgroundColor = mainHandles.colors{3,4};
handles.save.ForegroundColor = mainHandles.colors{3,2};
% Cancel button
handles.cancel.BackgroundColor = mainHandles.colors{2,4};
handles.cancel.ForegroundColor = mainHandles.colors{2,2};

% Update handles structure
guidata(hObject, handles);

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

figureHandle = findobj(0, 'tag', 'jobQueue');
mainHandles = guidata(figureHandle);

% Validate new configurations
% numIterations
numIterations = str2double(handles.numIterations.String);
if isnan(numIterations) || ...
        numIterations < 0 || ...
        rem(numIterations,1) ~= 0
    errordlg('Number of Iterations must be a natural number', 'Invalid Parameter');
    return;
end

% stripHeight
stripHeight = str2double(handles.stripHeight.String);
if isnan(stripHeight) || ...
        stripHeight < 0 || ...
        rem(stripHeight,1) ~= 0
    errordlg('Strip Height must be a natural number', 'Invalid Parameter');
    return;
end

% stripWidth
stripWidth = str2double(handles.stripWidth.String);
if isnan(stripWidth) || ...
        stripWidth < 0 || ...
        rem(stripWidth,1) ~= 0
    errordlg('Strip Width must be a natural number', 'Invalid Parameter');
    return;
end

% samplingRate
samplingRate = str2double(handles.samplingRate.String);
if isnan(samplingRate) || ...
        samplingRate < 0
    errordlg('Sampling Rate must be a positive real number', 'Invalid Parameter');
    return;
end

% minPeakRatio
minPeakRatio = str2double(handles.minPeakRatio.String);
if isnan(minPeakRatio) || ...
        minPeakRatio < 0
    errordlg('Minimum Peak Ratio must be a postive real number', 'Invalid Parameter');
    return;
end

% minPeakThreshold
minPeakThreshold = str2double(handles.minPeakThreshold.String);
if isnan(minPeakThreshold) || ...
        minPeakThreshold < 0
    errordlg('Minimum Peak Threshold must be a postive real number', 'Invalid Parameter');
    return;
end

% scalingFactor
scalingFactor = str2double(handles.scalingFactor.String);
if isnan(scalingFactor) || ...
        scalingFactor < 0
    errordlg('Scaling Factor must be a postive real number', 'Invalid Parameter');
    return;
end

% searchWindowHeight
searchWindowHeight = str2double(handles.searchWindowHeight.String);
if isnan(searchWindowHeight) || ...
        searchWindowHeight < 0 || ...
        rem(searchWindowHeight,1) ~= 0
    errordlg('Search Window Height must be a natural number', 'Invalid Parameter');
    return;
end

% neighborhoodSize
neighborhoodSize = str2double(handles.neighborhoodSize.String);
if isnan(neighborhoodSize) || ...
        neighborhoodSize < 0 || ...
        rem(neighborhoodSize,1) ~= 0
    errordlg('Neighborhood Size must be a natural number', 'Invalid Parameter');
    return;
end

% subpixelDepth
subpixelDepth = str2double(handles.subpixelDepth.String);
if isnan(subpixelDepth) || ...
        subpixelDepth < 0
    errordlg('Subpixel Depth must be a positive real number', 'Invalid Parameter');
    return;
end

% Save new configurations
mainHandles.config.fineOverwrite = logical(handles.overwrite.Value);
mainHandles.config.fineVerbosity = logical(handles.verbosity.Value);
mainHandles.config.fineNumIterations = str2double(handles.numIterations.String);
mainHandles.config.fineStripHeight = str2double(handles.stripHeight.String);
mainHandles.config.fineStripWidth = str2double(handles.stripWidth.String);
mainHandles.config.fineSamplingRate = str2double(handles.samplingRate.String);
mainHandles.config.fineMinPeakRatio = str2double(handles.minPeakRatio.String);
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



function minPeakRatio_Callback(hObject, eventdata, handles)
% hObject    handle to minPeakRatio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minPeakRatio as text
%        str2double(get(hObject,'String')) returns contents of minPeakRatio as a double


% --- Executes during object creation, after setting all properties.
function minPeakRatio_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minPeakRatio (see GCBO)
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