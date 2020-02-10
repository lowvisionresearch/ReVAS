function params = ModifyParams(inputArgument, varargin)
% params = ModifyParams(inputArgument, varargin)
%
%   Generic tool for creating sub-GUIs for parameter selection, adjustment,
%   and editing. For a given callerStr, it gets parameter fields, default
%   values, and validation functions from GetDefaults function, removes
%   the fields that cannot be set by the user, and creates a GUI with
%   uicontrols for each remaining parameter. 
% 
%   inputArgument must either be a char array, name of the module, e.g.,
%   StripAnalysis, or a cell array with a length of two
%   {callerStr,default}
%
%   Mehmet N. Agaoglu 1/19/2020 mnagaoglu@gmail.com

fprintf('%s: ModifyParams launched!\n',datestr(datetime));

if nargin < 1 
    error('ModifyParams needs at least one argument (callerStr)!');
end


% to allow for non-UI calls, we have the optional flag noGUI. If it is set
% to true, we get params and return.
if nargin < 2
    noGUI = 0;
else
    noGUI = varargin{1};
end

% if provided, this function draws all uicontrols within a parent graphics
% object.
if nargin < 3
    parent = [];
else
    parent = varargin{2};
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Get default params and their validation functions. 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get default parameters and validation functions
if ischar(inputArgument)
    callerStr = inputArgument;
    [default, validate] = GetDefaults(callerStr);  
elseif iscell(inputArgument) && length(inputArgument) == 2
    callerStr = inputArgument{1};
    default = inputArgument{2};
    [~, validate] = GetDefaults(callerStr);
else
    errStr = ['ModifyParams: callerStr must either be a char array, name of the module'...
           'e.g., StripAnalysis, or a cell array with a length of two '...
           '{callerStr,default,validate}'];
    errordlg(errStr,'ModifyParams error','modal');
    fprintf('%s: ModifyParams returned with an error: %s\n',datestr(datetime),errStr);
end

% in case user closes the GUI window, return default values
params = default;

% parameter names
paramNames = fieldnames(default);

% remove field that user cannot modify via GUI, e.g., axesHandles
toRemove = contains(paramNames, 'axesHandles')    | ...
           contains(paramNames, 'referenceFrame') | ...
           contains(paramNames, 'peakValues')     | ...
           contains(paramNames, 'position')       | ...
           contains(paramNames, 'timeSec')        | ...
           contains(paramNames, 'rowNumbers')     | ...
           contains(paramNames, 'oldStripHeight') | ...
           contains(paramNames, 'badFrames')      | ...
           contains(paramNames, 'trim')           | ...
           contains(paramNames, 'tilts')          | ...
           contains(paramNames, 'toneCurve');
params = rmfield(params,paramNames(toRemove));
paramNames(toRemove) = [];

if noGUI
    return;
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Determine parameter types. UI controls will be created accordingly.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% first look for logical or string type params, and also get tooltip strings
numOfParams = length(paramNames);
logicalParams = false(numOfParams,1);
strParams = false(numOfParams,1);
multiTypeParams = false(numOfParams,1);
tooltips = cell(numOfParams,1);
characterLength = nan(numOfParams,1);
for i=1:numOfParams
    
    % tooltips show the validation functions
    tooltips{i} = func2str(validate.(paramNames{i}));
    
    % character length will be used to properly position uicontrols
    characterLength(i) = length(paramNames{i});
    
    % label params as logical(checkbox) or string(popup menu)
    logicalParams(i) = contains(tooltips{i},'islogical');
    strParams(i) = ischar(default.(paramNames{i}));
    
    % some parameters accept filepath, array, or number. Treat them as edit
    % boxes.
    multiTypeParams(i) = logicalParams(i) & contains(tooltips{i},'ischar') & ...
        isempty(default.(paramNames{i}));

end
% make sure we remove duplicate classification of parameter type
logicalParams(multiTypeParams) = false;

% now get different options for string type params.
strParamsIndex = find(strParams);
options = cell(sum(strParams),1);
for i=1:length(options)
    strToEvaluate = regexp(tooltips{strParamsIndex(i)},'{.*}','match');
    options{i} = eval(strToEvaluate{1});
end

% the rest of the params will assumed to be numeric.
numericParams = ~logicalParams & ~strParams;

% read the documentation for the callerStr module.
helpStr = ReadHelp(callerStr);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Create GUI figure
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get display dimensions and place the gui to the right edge of the display
if isempty(parent)
    r = groot;
    screenSize = r.ScreenSize;
    ppi = r.ScreenPixelsPerInch;
    fontSize = 12;
else
    screenSize = parent.UserData.screenSize;
    ppi = parent.UserData.ppi;
    fontSize = parent.UserData.fontSize;
end
guiSize = [min(screenSize(3), max(characterLength)*3*ppi/fontSize)...
           min(screenSize(4), (numOfParams + 2)*fontSize*2.5)];
rowSize = guiSize(2)/(numOfParams + 2);

% if a parent graphics object is provided use it. Otherwise, get screen
% dimensions and put the GUI to the top-right corner
if ~isempty(parent)
    parentPos = get(parent,'position');
    guiPos = [sum(parentPos([1 3])) sum(parentPos([2 4]))-guiSize(2) guiSize];
    if sum(guiPos([1 3])) > screenSize(3)
        guiPos(1) = screenSize(3)-guiSize(1);
    end
    if sum(guiPos([2 4])) > screenSize(4)
        guiPos(2) = screenSize(3)-guiSize(4);
    end
else
    guiPos = [screenSize(3:4)-guiSize guiSize];
end

% create the figure but keep it invisible while populating with uicontrols.
gui = struct;
gui.ui = struct;
gui.fig = figure('units','pixels',...
    'position',guiPos,...
    'menubar','none',...
    'toolbar','none',...
    'name',[callerStr ' Parameters'],...
    'numbertitle','off',...
    'resize','off',...
    'visible','off',...
    'closerequestfcn',{@CancelCallback},...
    'windowstyle','modal');

% see if we have a suitable GPU. If not, we will disable relevant controls
nGPU = gpuDeviceCount; 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Populate the gui with uicontrols
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% start with the logical params
logicalParamsIndices = find(logicalParams);
logicalParamsNames = flipud(sort(paramNames(logicalParamsIndices))); %#ok<FLPST>
maxCharLength = max(characterLength(logicalParamsIndices));

for i=1:length(logicalParamsIndices)

    % location of the current uicontrol
    yLoc = guiSize(2) - (i)*rowSize;
    xLoc = (guiSize(1) - maxCharLength * ppi/fontSize) /2;
    thisPosition = [xLoc yLoc guiSize(1) rowSize];
    
    % disable enableGPU field if there is no available GPU
    enable = 'on';
    if strcmp(logicalParamsNames{i},'enableGPU')
        if nGPU == 0
            enable = 'off';
        end
    end
    
    % create the uicontrol
    gui.ui.((logicalParamsNames{i})) = ...
        uicontrol(gui.fig,...
        'style','checkbox',...
        'units','pix',...
        'position',thisPosition,...
        'fontsize',fontSize,...
        'string',logicalParamsNames{i},...
        'value',double(default.(logicalParamsNames{i})),...
        'tooltip',tooltips{logicalParamsIndices(i)},...
        'callback',{@LogicalCallback,validate.(logicalParamsNames{i})},...
        'enable',enable);
end


% next, take care of string params
strParamsIndices = find(strParams);
for i=1:length(strParamsIndices)

    % location of the current uicontrol
    yLoc = guiSize(2) - (i + length(logicalParamsIndices)) * rowSize;
    labelPos = [1 yLoc-rowSize/4 guiSize(1)*0.35 rowSize];
    thisPosition = [guiSize(1)*0.4 labelPos(2)+rowSize/4 guiSize(1)*0.45 labelPos(4)];
    
    % value of default string
    thisDefault = default.(paramNames{strParamsIndices(i)});
    thisValue = find(contains(options{i}, thisDefault));
    
    % remove CUDA option if there is no available GPU
    if strcmp(paramNames{strParamsIndices(i)},'corrMethod')
        if nGPU == 0
            ix = contains(options{i},'cuda');
            options{i}(ix) = [];
        end
    end
    
    % create the uicontrol
    fld = (paramNames{strParamsIndices(i)});
    gui.ui.(fld) = ...
        uicontrol(gui.fig,...
        'style','popupmenu',...
        'units','pix',...
        'position',thisPosition,...
        'fontsize',fontSize,...
        'string',options{i},...
        'value',thisValue,...
        'tag',paramNames{strParamsIndices(i)},...
        'tooltip',tooltips{strParamsIndices(i)},...
        'callback',{@StringCallback,validate.(paramNames{strParamsIndices(i)})});
    
    % create the label
    gui.ui.([fld 'Label']) = ...
        uicontrol(gui.fig,...
        'style','text',...
        'units','pix',...
        'position',labelPos,...
        'fontsize',fontSize,...
        'horizontalalignment','right',...
        'string',paramNames{strParamsIndices(i)});
    
    % now align text/edit box pairs
    align([gui.ui.(fld); gui.ui.([fld 'Label'])],'distribute','center');

end

% Layout all numeric fields. 
% Note that for every numeric field, we have to put a static text field as
% its label.
numericParamsIndices = find(numericParams);
yStart = length(logicalParamsIndices) + length(strParamsIndices);

for i=1:length(numericParamsIndices)
    
    % location of the current uicontrol
    yLoc = guiSize(2) - (i + yStart) * rowSize;
    labelPos = [1 yLoc-rowSize/4 guiSize(1)*0.55 rowSize];
    thisPosition = [guiSize(1)*0.6 labelPos(2)+rowSize/4 guiSize(1)*0.3 labelPos(4)];
    
    % create the uicontrol
    fld = (paramNames{numericParamsIndices(i)});
    gui.ui.(fld) = ...
        uicontrol(gui.fig,...
        'style','edit',...
        'units','pix',...
        'position',thisPosition,...
        'fontsize',fontSize,...
        'string',num2str(default.(paramNames{numericParamsIndices(i)})),...
        'tooltip',tooltips{numericParamsIndices(i)},...
        'tag',paramNames{numericParamsIndices(i)},...
        'callback',{@NumericCallback,validate.(paramNames{numericParamsIndices(i)})});
    
    % create the label
    gui.ui.([fld 'Label']) = ...
        uicontrol(gui.fig,...
        'style','text',...
        'units','pix',...
        'position',labelPos,...
        'fontsize',fontSize,...
        'horizontalalignment','right',...
        'string',paramNames{numericParamsIndices(i)});
    
    % now align text/edit box pairs
    align([gui.ui.(fld); gui.ui.([fld 'Label'])],'distribute','center');

end

% also before we finish up, add the OK and Cancel buttons.
okPos = [guiSize(1)*0.15 guiSize(2)*(0.5/(numOfParams+1)) guiSize(1)*0.3 rowSize];
cancelPos = [guiSize(1)*0.55 okPos(2:4)];

% OK button
gui.ui.OK = uicontrol(gui.fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',okPos,...
    'fontsize',fontSize,...
    'string','OK',...
    'callback',{@OkCallback});
    
% Cancel button
gui.ui.Cancel = uicontrol(gui.fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',cancelPos,...
    'fontsize',fontSize,...
    'string','Cancel',...
    'callback',{@CancelCallback});
    
% help/info button
load([ipticondir filesep 'help_icon.mat'],'helpIcon');
[m,n,~] = size(helpIcon);
gui.ui.Help = uicontrol(gui.fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',[guiSize-rowSize m n],...
    'fontsize',fontSize,...
    'cdata',helpIcon,...
    'string','',...
    'tooltip','Help',...
    'callback',{@GetHelp}); 
    
% Make visible 
set(gui.fig,'visible','on');

% prevent all other processes from starting until closed.
uiwait(gui.fig);        



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Callback functions
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function LogicalCallback(src,~,varargin)
        params.(src.String) = src.Value;
    end

    function StringCallback(src,~,varargin)
        if ~(strcmp(src.Tag, 'enableVerbosity'))
            params.(src.Tag) = src.String{src.Value};
        else
            params.(src.Tag) = src.Value;
        end
    end

    function NumericCallback(src,~,varargin)
        value = str2num(['[' src.String ']']); %#ok<ST2NM>
        validateFunc = varargin{1};
        
        if ~validateFunc(value)
            src.BackgroundColor = [1 .5 .5];
            pause(0.5);
            src.BackgroundColor = 0.94 * [1 1 1];
            src.String = num2str(params.(src.Tag));
        else
            params.(src.Tag) = value;
        end
    end

    function OkCallback(varargin)
        fprintf('%s: ModifyParams, user clicked OK.\n',datestr(datetime));
        delete(gui.fig);
    end

    % If user clicks Cancel, return an empty array.
    function CancelCallback(varargin)
        params = [];
        fprintf('%s: ModifyParams, user cancelled.\n',datestr(datetime));
        delete(gui.fig); 
    end
    
    function GetHelp(varargin)
        fprintf('%s\n',helpStr);
        helpSize = screenSize(3:4)/2;
        helpPos = [screenSize(3:4)/4 helpSize];
        
        helpFig = figure('units','pixels',...
            'menubar','none',...
            'toolbar','none',...
            'position',helpPos,...
            'windowstyle','modal',...
            'name',[callerStr ' Help'],...
            'numbertitle','off',...
            'resize','off',...
            'visible','off',...
            'windowstyle','modal');
        
        textField = uicontrol(helpFig,...
            'style','edit',...
            'units','pixels',...
            'min',0,...
            'max',3,...
            'enable','inactive',...
            'HorizontalAlign','left',...
            'position',[1 1 helpPos(3:4)],...
            'fontsize',fontSize,...
            'string',helpStr); %#ok<NASGU>
        
        set(helpFig,'visible','on') 
        
    end

end



