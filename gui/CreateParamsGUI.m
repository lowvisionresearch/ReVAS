function params = CreateParamsGUI(callerStr)

if nargin < 1 
    callerStr = 'RemoveStimuli';
    % error('CreateParamsGUI needs at least one argument (callerStr)!');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Get default params and their validation functions. 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get default parameters and validation functions
[default, validate] = GetDefaults(callerStr);

% in case user closes the GUI window, return default values
params = default;

% parameter names
paramNames = fieldnames(default);

% remove field that user cannot modify via GUI, e.g., axesHandles
paramNames(contains(paramNames, 'axesHandles')) = [];
paramNames(contains(paramNames, 'referenceFrame')) = [];
paramNames(contains(paramNames, 'peakValues')) = [];
paramNames(contains(paramNames, 'position')) = [];
paramNames(contains(paramNames, 'timeSec')) = [];
paramNames(contains(paramNames, 'rowNumbers')) = [];
paramNames(contains(paramNames, 'oldStripHeight')) = [];
paramNames(contains(paramNames, 'badFrames')) = [];
paramNames(contains(paramNames, 'trim')) = [];
paramNames(contains(paramNames, 'tilts')) = [];
paramNames(contains(paramNames, 'toneCurve')) = [];
numOfParams = length(paramNames);

% first look for logical or string type params, and also get tooltip strings
logicalParams = false(numOfParams,1);
strParams = false(numOfParams,1);
tooltips = cell(numOfParams,1);
characterLength = nan(numOfParams,1);
for i=1:numOfParams
    tooltips{i} = func2str(validate.(paramNames{i}));
    characterLength(i) = length(paramNames{i});
    logicalParams(i) = contains(tooltips{i},'islogical');
    strParams(i) = ischar(default.(paramNames{i}));
end

% now get different options for string type params.
strParamsIndex = find(strParams);
options = cell(sum(strParams));
for i=1:length(options)
    strToEvaluate = regexp(tooltips{strParamsIndex(i)},'{.*}','match');
    options{i} = eval(strToEvaluate{1});
end

% the rest of the params will assumed to be numeric.
numericParams = ~logicalParams & ~strParams;

% read the documentation for the callerStr module.
filepath = [FindFile(callerStr) '.m'];
text = fileread(filepath);
expression = '(\n{2}?)|(%{2}?)';
matchStr = regexp(text,expression);
helpStr = text(1:matchStr(1));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Create GUI figure
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get display dimensions and place the gui to the right edge of the display
r = groot;
screenSize = r.ScreenSize;
ppi = r.ScreenPixelsPerInch;
fontSize = 12;
guiSize = [min(screenSize(3), max(characterLength)*3*ppi/fontSize)...
           min(screenSize(4), (numOfParams + 1)*fontSize*2.5)];
rowSize = guiSize(2)/(numOfParams + 1);

% create the figure but keep it invisible while populating with uicontrols.
gui = struct;
gui.ui = struct;
gui.fig = figure('units','pixels',...
    'position',[screenSize(3:4)-guiSize guiSize],...
    'menubar','none',...
    'toolbar','none',...
    'name',[callerStr ' Parameters'],...
    'numbertitle','off',...
    'resize','off',...
    'visible','on',...
    'closerequestfcn',{@CancelCallback});


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
    yLoc = guiSize(2) - (i-1)*rowSize;
    xLoc = (guiSize(1) - maxCharLength * ppi/fontSize) /2;
    thisPosition = [xLoc yLoc guiSize(1) rowSize];
    
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
        'callback',{@LogicalCallback,validate.(logicalParamsNames{i})});
end


% next, take care of string params
strParamsIndices = find(strParams);
for i=1:length(strParamsIndices)

    % location of the current uicontrol
    yLoc = guiSize(2) - (i-1 + length(logicalParamsIndices)) * rowSize;
    thisPosition = [guiSize(1)*0.2 yLoc guiSize(1)*0.6 rowSize];
    
    % value of default string
    thisDefault = default.(paramNames{strParamsIndices(i)});
    thisValue = find(contains(options{i}, thisDefault));
    
    % create the uicontrol
    gui.ui.(paramNames{strParamsIndices(i)}) = ...
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
end

% Finally, layout all numeric fields. 
% Note that for every numeric field, we have to put a static text field as
% its label.
numericParamsIndices = find(numericParams);
yStart = length(logicalParamsIndices) + length(strParamsIndices);

for i=1:length(numericParamsIndices)
    
    % location of the current uicontrol
    yLoc = guiSize(2) - (i-1 + yStart) * rowSize;
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
okPos = [guiSize(1)*0.15 guiSize(2)*(0.5/(numOfParams+1)) ...
    guiSize(1)*0.3 guiSize(2)/(numOfParams+1)];
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
        params.(src.Tag) = src.String{src.Value};
    end

    function NumericCallback(src,~,varargin)
        value = str2double(src.String);
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
        delete(gui.fig);
    end

    % If user clicks Cancel, return an empty array.
    function CancelCallback(varargin)
        params = [];
        delete(gui.fig); 
    end
    
    function GetHelp(varargin)
        format compact;
        disp(helpStr)
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
            'visible','off');
        
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
        uiwait(helpFig);  
        
    end

end



