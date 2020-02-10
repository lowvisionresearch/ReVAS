function PipelineTool(varargin)
% PipelineTool(varargin)
%
%   A tool to create a new processing pipeline using functions under
%   ReVAS/corepipeline folder.
%
% Mehmet N. Agaoglu 1/19/2020 

fprintf('%s: PipelineTool launched!\n',datestr(datetime));

% first argument is the source
src = varargin{1};

% get siblings 
siblingObjs = get(get(src,'parent'),'children');

% keep track of changes
isChange = false;

% the third argument is the handle from main gui
revas = varargin{3};
screenSize = revas.screenSize;
fontSize = revas.fontSize;
parentPos = revas.gui.OuterPosition;

% create the GUI
guiSize = round([min(screenSize(3)*0.5, 480) min(screenSize(4)*0.5, 360)]);
fig = figure('units','pixels',...
    'position',[parentPos(1:2)+(parentPos(3:4)-guiSize)/2 guiSize],...
    'menubar','none',...
    'toolbar','none',...
    'name','New Pipeline Creation Tool',...
    'numbertitle','off',...
    'resize','off',...
    'visible','off',...
    'windowstyle','modal');

% check all available core modules
corePath = fileparts(which('StripAnalysis'));
modules = dir([corePath filesep '*.m']);

% make sure all modules have their default params and validation functions
% defined.
missingModules = [];
moduleNames = {};
for i=1:length(modules)
    % remove the file extension
    fileName = modules(i).name;
    fileName(strfind(fileName,'.m'):end) = [];
    
    % try getting defaults. if the module is not defined in GetDefaults,
    % this will fail and we'll take a note of that and let the user know.
    try
        GetDefaults(fileName);
    catch
        missingModules = [missingModules; i]; %#ok<AGROW>
    end
    moduleNames = [moduleNames; {fileName}]; %#ok<AGROW>
end
if ~isempty(missingModules)
    warStr = 'NewPipeline found some modules in corepipeline with missing entries in GetDefaults.m';
    warndlg(warStr,'Missing info','modal');
    fprintf('%s: PipelineTool, %s\n',datestr(datetime),warStr);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add uicontrols
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we need two listboxes. One for listing available modules, and another one
% for showing the created pipeline.

% Available modules listbox
availablePos = [5 75 guiSize(1)/2.5 guiSize(2)-120];
lbAvailable = ...
    uicontrol(fig,...
    'style','listbox',...
    'units','pix',...
    'position',availablePos,...
    'fontsize',fontSize,...
    'string',moduleNames,...
    'value',1,...
    'tooltip','List of available corepipeline modules.');

% Available modules label
labelPos = [availablePos(1) availablePos(2)+availablePos(4)+5 availablePos(3) 30];
lbAvailableLabel = ...
    uicontrol(fig,...
    'style','text',...
    'units','pix',...
    'position',labelPos,...
    'fontsize',fontSize,...
    'fontweight','bold',...
    'horizontalalignment','center',...
    'string','Available Modules'); %#ok<NASGU>

% pipeline listbox
pipePos = [guiSize(1)-availablePos(3)-5 75 availablePos(3:4)];
lbPipe = ...
    uicontrol(fig,...
    'style','listbox',...
    'units','pix',...
    'position',pipePos,...
    'fontsize',fontSize,...
    'string',{},...
    'value',0,...
    'tooltip',['Modules will be executed in the order shown here.' newline...
               'Double click to edit parameters of a module.'],...
    'callback',{@EditParamsCallback,revas});

% populate it if revas already has a pipeline AND edit pipeline was pressed
if strcmp(src.Text,'Edit')
    lbPipe.String = revas.gui.UserData.pipeline;
    lbPipe.Value = length(lbPipe.String);
end

% pipeline label
pipeLabelPos = [pipePos(1) pipePos(2)+pipePos(4)+5 pipePos(3) 30];
lbAvailableLabel = ...
    uicontrol(fig,...
    'style','text',...
    'units','pix',...
    'position',pipeLabelPos,...
    'fontsize',fontSize,...
    'fontweight','bold',...
    'horizontalalignment','center',...
    'string','Pipeline'); %#ok<NASGU>

% add module to pipeline button
d = min(guiSize(1:2))*0.05;
addPos = [guiSize(1:2)/2 - d*[1 1] d*2*[1 1]];
addModule = uicontrol(fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',addPos,...
    'fontsize',fontSize,...
    'string','->',...
    'callback',{@AddCallback}); %#ok<NASGU>

% remove module from pipeline button
removePos = [addPos(1) addPos(2)+addPos(4)+10 addPos(3:4)];
removeModule = uicontrol(fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',removePos,...
    'fontsize',fontSize,...
    'string','<-',...
    'callback',{@RemoveCallback}); %#ok<NASGU>

% also before we finish up, add the OK and Cancel buttons.

% OK button
okPos = [guiSize(1)*0.15 5 guiSize(1)*0.3 40];
okButton = uicontrol(fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',okPos,...
    'fontsize',fontSize,...
    'string','OK',...
    'callback',{@OkCallback}); %#ok<NASGU>
    
% Cancel button
cancelPos = [guiSize(1)*0.55 okPos(2:4)];
cancelButton = uicontrol(fig,...
    'style','pushbutton',...
    'units','pix',...
    'position',cancelPos,...
    'fontsize',fontSize,...
    'string','Cancel',...
    'callback',{@CancelCallback}); %#ok<NASGU>

% Make visible 
set(fig,'visible','on');

% prevent all other processes from starting until closed.
uiwait(fig);  


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% callbacks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function AddCallback(varargin)
        % move selected modules to pipeline, add to the bottom of the list
        newModule = lbAvailable.String(lbAvailable.Value);
        lbPipe.String = [lbPipe.String; newModule];
        
        % make the last item in the pipeline as the selected item
        lbPipe.Value = length(lbPipe.String);
        
        % get defaults for the added module. 
        revas.gui.UserData.pipeParams{lbPipe.Value,1} = ModifyParams(newModule{1}, true);
    end

    function RemoveCallback(varargin)
        % if there is no module in the pipeline yet, return
        if lbPipe.Value < 1
            return;
        end
        
        % remove the corresponding params from revas
        revas.gui.UserData.pipeParams(lbPipe.Value) = [];
        
        % remove the selected module from pipeline
        lbPipe.String(lbPipe.Value) = [];
        
        % make the last item in the pipeline as the selected item
        lbPipe.Value = length(lbPipe.String);
    end

    function EditParamsCallback(varargin)
        % do this only when user double-clicks on an item
        if strcmp(get(fig,'SelectionType'),'open')
            
            % which module is selected?
            thisModule = lbPipe.String(lbPipe.Value);
            
            % call parameter GUI to adjust params. if output is not empty,
            % update the pipeParams.
            inp = {thisModule{1}, revas.gui.UserData.pipeParams{lbPipe.Value}};
            newParams = ModifyParams(inp, false, fig);
            if ~isempty(newParams)
                oldParams = revas.gui.UserData.pipeParams{lbPipe.Value,1};
                isChange = CompareFieldsHelper(oldParams,newParams);
                if isChange
                    revas.gui.UserData.pipeParams{lbPipe.Value,1} = newParams;
                end
            end
        end
    end

    function OkCallback(varargin)
        
        % if pipeline changed, enable Save, Edit, and Save As menus.
        if isChange
            % enable save and saveas menus
            set(siblingObjs(~contains({siblingObjs.Text},{'New','Open'})),'Enable','on');
        end
        
        % assign the pipeline to output
        revas.gui.UserData.pipeline = lbPipe.String;
        
        % call the pipeline config function to populate/update the pipeline
        % panel. if pipeline is empty, disable
        pl = revas.gui.UserData.pipeline;
        revas.gui.UserData.lbPipeline.String = pl;
        revas.gui.UserData.lbPipeline.Value = length(pl);
        
        % disable pipe listbox if pipeline is empty
        if isempty(pl)
            revas.gui.UserData.lbPipeline.Visible = 'off';
        else
            revas.gui.UserData.lbPipeline.Visible = 'on';
        end
        
        % close the gui
        fprintf('%s: PipelineTool closed.\n',datestr(datetime));
        delete(fig);
    end

    % If user clicks Cancel, return an empty array.
    function CancelCallback(varargin)
        % close the gui
        fprintf('%s: PipelineTool cancelled.\n',datestr(datetime));
        delete(fig); 
    end


end