function ReVAS

% get version number from readme.md
versionNo = RevasVersion;

% log file name.
logFile = [fileparts(which('ReVAS')) filesep 'log.txt']; 

% if log file grows too big, rename it and open a new file.
if exist(logFile,'file')
    stats = dir(logFile);
    if stats.bytes > 10^8
        movefile(logFile,...
            [logFile(1:end-4) '-' regexprep(datestr(datetime),'[ :]','-') '.txt'])
    end
end
eval(['diary ' logFile]);
fprintf('\n\n%s: ReVAS %s launched!\n',datestr(datetime), versionNo)

% set the abort level
global abortTriggered;
abortTriggered = false;

% to keep track of user changes to UI
isChange = false;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create GUI figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get display dimensions and place the gui to the right edge of the display
r = groot;
revas.ppi = r.ScreenPixelsPerInch;
revas.screenSize = r.ScreenSize;
revas.fontSize = 12;
revas.ppi = r.ScreenPixelsPerInch;

% create gui
guiSize = [min(revas.screenSize(3)*0.5,640) min(revas.screenSize(4)*0.5,480)];
revasPos = [(revas.screenSize(3:4)-guiSize)/2 guiSize];
revas.gui = figure('units','pixels',...
    'position',revasPos,...
    'menubar','none',...
    'toolbar','none',...
    'name',['ReVAS ' versionNo],...
    'numbertitle','off',...
    'resize','on',...
    'visible','on',...
    'closerequestfcn',{@RevasClose});

% save some of the config under UserData so that other child GUIs can also
% use them.
revas.gui.UserData.screenSize = revas.screenSize;
revas.gui.UserData.fontSize = revas.fontSize;
revas.gui.UserData.ppi = revas.ppi;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create menus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% video menu
revas.pipelineMenu = uimenu(revas.gui,'Text','File');
uimenu(revas.pipelineMenu,'Text','Raw','MenuSelectedFcn',{@RevasFileSelect,{'.avi'},{'trim','nostim','gammscaled','bandfilt'},revas});
uimenu(revas.pipelineMenu,'Text','Trimmed','MenuSelectedFcn',{@RevasFileSelect,{'_trim.avi'},{},revas});
uimenu(revas.pipelineMenu,'Text','Stimulus Removed','MenuSelectedFcn',{@RevasFileSelect,{'_nostim.avi'},{},revas});
uimenu(revas.pipelineMenu,'Text','Gamma Corrected','MenuSelectedFcn',{@RevasFileSelect,{'_gammscaled.avi'},{},revas});
uimenu(revas.pipelineMenu,'Text','Bandpass Filtered','MenuSelectedFcn',{@RevasFileSelect,{'_bandfilt.avi'},{},revas});
uimenu(revas.pipelineMenu,'Text','Eye Position','MenuSelectedFcn',{@RevasFileSelect,{'_position.mat','_filtered.mat'},{},revas},'Separator','on');

% pipeline menu
revas.pipelineMenu = uimenu(revas.gui,'Text','Pipeline');
revas.gui.UserData.new = uimenu(revas.pipelineMenu,'Text','New','Accelerator','N','MenuSelectedFcn',{@PipelineTool,revas});
revas.gui.UserData.open = uimenu(revas.pipelineMenu,'Text','Open','Accelerator','O','MenuSelectedFcn',{@OpenPipeline,revas});
revas.gui.UserData.edit = uimenu(revas.pipelineMenu,'Text','Edit','Accelerator','E','MenuSelectedFcn',{@PipelineTool,revas},'Enable','of');
revas.gui.UserData.save = uimenu(revas.pipelineMenu,'Text','Save','Accelerator','S','MenuSelectedFcn',{@SavePipeline,revas},'Enable','of');
revas.gui.UserData.saveas = uimenu(revas.pipelineMenu,'Text','Save As','Accelerator','A','MenuSelectedFcn',{@SaveAsPipeline,revas},'Enable','of');

% help menu
revas.helpMenu = uimenu(revas.gui,'Text','Help');
uimenu(revas.helpMenu,'Text','About','Accelerator','I','MenuSelectedFcn',{@RevasAbout,revas});
uimenu(revas.helpMenu,'Text','Report An Issue','MenuSelectedFcn',{@RevasReportIssue,revas});
uimenu(revas.helpMenu,'Text','Documentation','Accelerator','D','MenuSelectedFcn',{@RevasDocumentation,revas});

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Create uicontrols
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file selection panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filePanelPos = [0 0.4 3/8 0.6];
revas.gui.UserData.filePanel = uipanel('Parent',revas.gui,...
             'units','normalized',...
             'Title','File Selection',...
             'FontSize',revas.fontSize,...
             'position',filePanelPos,...
             'tag','filePanel');   
         
revas.gui.UserData.fileMessage = uicontrol(revas.gui.UserData.filePanel,...
             'style','text',...
             'units','normalized',...
             'position',[.1 .1 .8 .8],...
             'fontsize',revas.fontSize,...
             'fontweight','bold',...
             'horizontalalignment','center',...
             'string','Please select some videos or eye position files using File menu.');         

% file listbox
revas.gui.UserData.lbFile = uicontrol(revas.gui.UserData.filePanel,...
             'style','listbox',...
             'units','normalized',...
             'position',[0 0 1 1],...
             'fontsize',revas.fontSize,...
             'string',{},...
             'value',0,...
             'tooltip','Select files that you want to analyze.',...
             'callback',{@DoNothing,revas},...
             'visible','off',...
             'min',1,...
             'max',3);         
         
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pipeline panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pipePanelPos = [sum(filePanelPos([1 3])) filePanelPos(2:4)];
revas.gui.UserData.pipelinePanel = uipanel('Parent',revas.gui,...
             'units','normalized',...
             'Title','Pipeline',...
             'FontSize',revas.fontSize,...
             'position',pipePanelPos,...
             'tag','pipePanel');
         
revas.gui.UserData.pipeMessage = uicontrol(revas.gui.UserData.pipelinePanel,...
             'style','text',...
             'units','normalized',...
             'position',[.1 .1 .8 .8],...
             'fontsize',revas.fontSize,...
             'fontweight','bold',...
             'horizontalalignment','center',...
             'string','Please create a new pipeline, or open an existing one using Pipeline menu.'); 
         
% pipeline listbox
revas.gui.UserData.lbPipeline = uicontrol(revas.gui.UserData.pipelinePanel,...
             'style','listbox',...
             'units','normalized',...
             'position',[0 0 1 1],...
             'fontsize',revas.fontSize,...
             'string',{},...
             'value',0,...
             'tooltip',['Modules will be executed in the order shown here.' newline...
                        'Double click to edit parameters of a module.'],...
             'callback',{@EditParamsCallback,revas},...
             'visible','off');          
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% global flags panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
globalPanelPos = [sum(pipePanelPos([1 3])) pipePanelPos(2) ...
    (1-pipePanelPos(3)-filePanelPos(3)) pipePanelPos(4)];
revas.gui.UserData.globalPanel = uipanel('Parent',revas.gui,...
             'units','normalized',...
             'Title','Global Flags',...
             'FontSize',revas.fontSize,...
             'position',globalPanelPos,...
             'tag','globalPanel');   

% global flags are overwrite, enableVerbosity, enableGPU, enableParallel,
% inMemory, saveLog
gParams = struct;
gParams.overwrite = 1;
gParams.inMemory = 0;
gParams.enableVerbosity = 1;
gParams.enableGPU = 0;
gParams.enableParallel = 0;
gParams.saveLog = 1;
gParamsNames = fieldnames(gParams);
rowSize = 0.8 / length(gParamsNames);

for i=1:length(gParamsNames)

    % location of the current uicontrol
    yLoc = 0.9 - (i)*rowSize;
    xLoc = 0.2;
    thisPosition = [xLoc yLoc globalPanelPos(1) rowSize];
    
    % disable fields if there is no GPU or multiple logical cores
    enable = true;
    switch gParamsNames{i}
        case 'enableGPU'
            enable = gpuDeviceCount ~= 0;
        case 'enableParallel'
            enable = feature('numcores') > 1;
        otherwise 
            % do nothing
    end
    if enable
        enable = 'on';
    else
        enable = 'off';
    end
    
    % create the uicontrol
    revas.gui.UserData.((gParamsNames{i})) = ...
        uicontrol(revas.gui.UserData.globalPanel,...
        'style','checkbox',...
        'units','normalized',...
        'position',thisPosition,...
        'fontsize',revas.fontSize,...
        'string',gParamsNames{i},...
        'value',double(gParams.(gParamsNames{i})),...
        'callback',{@GParamsCallback},...
        'enable',enable);
end
         
         
         
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log, status, and run 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
logBoxPos = [0 0.1 1 pipePanelPos(2)-0.13];
revas.gui.UserData.logBox = uicontrol(revas.gui,...
             'style','listbox',...
             'units','normalized',...
             'position',logBoxPos,...
             'fontsize',revas.fontSize,...
             'string',{},...
             'value',0,...
             'tooltip','Log window.');  
         
runPos = [.45 0 0.1 logBoxPos(2)];
revas.gui.UserData.runButton = uicontrol(revas.gui,...
             'style','togglebutton',...
             'units','normalized',...
             'position',runPos,...
             'fontsize',revas.fontSize,...
             'string','Run',...
             'value',0,...
             'callback',{@RunPipeline}); 

revas.gui.UserData.statusBar = axes(revas.gui,...
             'units','normalized',...
             'position',[0 sum(logBoxPos([2,4])) 1 0.03],...
             'xtick',[],'ytick',[],'tag','statusBar');         

revas.gui.UserData.pipeline = {};
revas.gui.UserData.pipeParams = {};
revas.gui.UserData.fileList = {};
         
% Make visible 
set(revas.gui,'visible','on');

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % pipeline menu callbacks
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function RevasReportIssue(varargin)
        fprintf('\n\n%s: Launching web browser for filing an issue.\n',datestr(datetime))
        web('https://github.com/lowvisionresearch/ReVAS/issues', '-browser');
    end
    
    function RevasDocumentation(varargin)
        fprintf('\n\n%s: Launching web browser for ReVAS wiki.\n',datestr(datetime))
        web('https://github.com/lowvisionresearch/ReVAS/wiki', '-browser');
    end

    function EditParamsCallback(varargin)

        % do this only when user double-clicks on an item
        if strcmp(get(revas.gui,'SelectionType'),'open')
            
            % which module is selected?
            val = revas.gui.UserData.lbPipeline.Value;
            thisModule = revas.gui.UserData.lbPipeline.String(val);
            
            % call parameter GUI to adjust params. if output is not empty,
            % update the pipeParams.
            inp = {thisModule{1}, revas.gui.UserData.pipeParams{val}};
            newParams = ModifyParams(inp, false, revas.gui);
            if ~isempty(newParams)
                oldParams = revas.gui.UserData.pipeParams{val,1};
                
                % see if there was any change
                isChange = CompareFieldsHelper(oldParams,newParams);
                if isChange
                    revas.gui.UserData.pipeParams{val,1} = newParams;
                    fprintf('%s: New parameters for %s: \n',datestr(datetime), thisModule{1});
                    disp(newParams);
                end
            end
        end
        
        % if there is any change, enable save menus
        if isChange
            % enable save and saveas menus
            % get siblings 
            childrenObjs = get(revas.pipelineMenu,'children');
            set(childrenObjs(contains({childrenObjs.Text},{'Save','Save As'})),'Enable','on');
        end
    end

    function GParamsCallback(src,~,varargin)
        gParams.(src.String) = src.Value;
        
        switch src.String
            case 'enableGPU'
                gParams.enableParallel = ~src.Value & feature('numcores') > 1;
            case 'enableParallel'
                gParams.enableGPU = ~src.Value & gpuDeviceCount > 0;
            case 'saveLog'
                if src.Value
                    diary on;
                else
                    diary off;
                end
            otherwise
                % do nothing
        end
        
        fprintf('%s: Global Flags: \n',datestr(datetime));
        disp(gParams);
        RefreshGlobalFlagPanel;
    end

    function RefreshGlobalFlagPanel
        for fld=1:length(gParamsNames)
            thisField = gParamsNames{fld};
            revas.gui.UserData.(thisField).Value = double(gParams.(thisField));
        end
    end


    function RunPipeline(varargin)
    end


    %% close callback
    function RevasClose(varargin)
        
        % if there are unsaved changes to the pipeline, ask user if she
        % wants to save them.
        if isChange
            answer = questdlg('There are unsaved changes to the pipeline. What do you want to do?', ...
                'ReVAS Close: Unsaved changes to pipeline', ...
                'Exit w/o Saving','Cancel','Save & Exit','Save & Exit');
            % Handle response
            switch answer
                case 'Exit w/o Saving'
                    % do nothing
                case 'Cancel'
                    return;
                case 'Save & Exit'
                    SavePipeline(revas.gui.UserData.save,[],revas);
            end
        end
        fprintf('%s: ReVAS closed!\n',datestr(datetime))
        diary off;
        delete(revas.gui);
    end


end