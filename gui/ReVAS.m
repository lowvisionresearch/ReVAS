function ReVAS

% check if another instance of ReVAS is already open.
figHandles = findobj('Type', 'figure','tag','revasgui');
if ~isempty(figHandles)
    figure(figHandles)
    return;
end

% get version number from readme.md
versionNo = RevasVersion;

% get log file name and start diary
logFile = [fileparts(which('ReVAS')) filesep 'log.txt']; 
eval(['diary ' logFile]);

% name of the hidden file that keeps track of last used fileList
fileListFile = [fileparts(which('ReVAS')) filesep '.filelist.mat'];

% name of the hidden file that keeps track of last used pipeline
lastUsedPipelineFile = [fileparts(which('ReVAS')) filesep '.pipeline.mat'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create GUI figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get display dimensions and place the gui to the right edge of the display
r = groot;
revas.ppi = r.ScreenPixelsPerInch;
revas.screenSize = r.ScreenSize;
if ismac
    revas.fontSize = 12;
else
    revas.fontSize = 10;
end
revas.ppi = r.ScreenPixelsPerInch;

% create gui
guiSize = [min(revas.screenSize(3)*0.75,1080) min(revas.screenSize(4)*0.75,720)];
revasPos = [(revas.screenSize(3:4)-guiSize)/2 guiSize];
revas.gui = figure('units','pix',...
    'position',revasPos,...
    'menubar','none',...
    'toolbar','none',...
    'name',['ReVAS ' versionNo],...
    'numbertitle','off',...
    'resize','on',...
    'visible','on',...
    'tag','revasgui',...
    'closerequestfcn',{@RevasClose});

% save some of the config under UserData so that other child GUIs can also
% use them.
revas.gui.UserData.screenSize = revas.screenSize;
revas.gui.UserData.fontSize = revas.fontSize;
revas.gui.UserData.ppi = revas.ppi;
revas.gui.UserData.logFile = logFile;
revas.gui.UserData.fileListFile = fileListFile;
revas.gui.UserData.lastUsedPipelineFile = lastUsedPipelineFile;

% to keep track of user changes to pipeline
revas.gui.UserData.isChange = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create menus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% video menu
revas.fileMenu = uimenu(revas.gui,'Text','File');
uimenu(revas.fileMenu,'Text','Raw','MenuSelectedFcn',{@RevasFileSelect,{'.avi'},{'trim','nostim','gammscaled','bandfilt'},revas});
uimenu(revas.fileMenu,'Text','Trimmed','MenuSelectedFcn',{@RevasFileSelect,{'_trim.avi'},{},revas});
uimenu(revas.fileMenu,'Text','Stimulus Removed','MenuSelectedFcn',{@RevasFileSelect,{'_nostim.avi'},{},revas});
uimenu(revas.fileMenu,'Text','Gamma Corrected','MenuSelectedFcn',{@RevasFileSelect,{'_gammscaled.avi'},{},revas});
uimenu(revas.fileMenu,'Text','Bandpass Filtered','MenuSelectedFcn',{@RevasFileSelect,{'_bandfilt.avi'},{},revas});
uimenu(revas.fileMenu,'Text','Eye Position','MenuSelectedFcn',{@RevasFileSelect,{'_position.mat','_filtered.mat'},{},revas},'Separator','on');
uimenu(revas.fileMenu,'Text','In Memory','MenuSelectedFcn',{@RevasFileSelect,{'_inmemory.mat'},{},revas},'Separator','on');
revas.gui.UserData.lastselectedfiles = uimenu(revas.fileMenu,'Text','Last Used','Accelerator','F','MenuSelectedFcn',{@RevasFileSelect,{},{},revas},'Separator','on','Enable',OnOffUtil(exist(fileListFile,'file')));

% pipeline menu
revas.pipelineMenu = uimenu(revas.gui,'Text','Pipeline');
uimenu(revas.pipelineMenu,'Text','New','Accelerator','N','MenuSelectedFcn',{@PipelineTool,revas});
uimenu(revas.pipelineMenu,'Text','Open','Accelerator','O','MenuSelectedFcn',{@OpenPipeline,revas,0});
uimenu(revas.pipelineMenu,'Text','Edit','Accelerator','E','MenuSelectedFcn',{@PipelineTool,revas},'Enable','off');
uimenu(revas.pipelineMenu,'Text','Save','Accelerator','S','MenuSelectedFcn',{@SavePipeline,revas},'Enable','off');
uimenu(revas.pipelineMenu,'Text','Save As','Accelerator','A','MenuSelectedFcn',{@SaveAsPipeline,revas},'Enable','off');
revas.gui.UserData.lastusedpipe = uimenu(revas.pipelineMenu,'Text','Last Used','Accelerator','P','MenuSelectedFcn',{@OpenPipeline,revas,1},'Separator','on','Enable',OnOffUtil(exist(lastUsedPipelineFile,'file')));

% run menu
revas.runMenu = uimenu(revas.gui,'Text','Run');
uimenu(revas.runMenu,'Text','Selected files','Accelerator','R','MenuSelectedFcn',{@RunMenuCallback,1});
uimenu(revas.runMenu,'Text','All files','MenuSelectedFcn',{@RunMenuCallback,0});

% help menu
revas.helpMenu = uimenu(revas.gui,'Text','Help');
uimenu(revas.helpMenu,'Text','About','Accelerator','I','MenuSelectedFcn',{@RevasAbout,revas});
uimenu(revas.helpMenu,'Text','Report An Issue','MenuSelectedFcn',{@RevasReportIssue,revas});
uimenu(revas.helpMenu,'Text','Documentation','Accelerator','D','MenuSelectedFcn',{@RevasDocumentation,revas});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Create uicontrols
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% layout
runPos = [.45 0 0.1 0.05]; 
abortPos = runPos;
logBoxPos = [0 0.05 1 0.125];
statusPos = [0 sum(logBoxPos([2 4])) 1 0.025];
globalPanelPos = [0 sum(statusPos([2 4])) 0.15 0.25];
pipePanelPos = [0 sum(globalPanelPos([2 4])) 0.15 0.30];
filePanelPos = [0 sum(pipePanelPos([2 4])) 0.15 0.25];
visualizePanelPos = [0.15 0.2 0.85 0.8];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visualization panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
revas.gui.UserData.visualizePanel = uipanel('Parent',revas.gui,...
             'units','normalized',...
             'Title','Visualization',...
             'FontSize',revas.fontSize,...
             'position',visualizePanelPos,...
             'tag','visualizePanel');  
         
revas.gui.UserData.imAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[0 .5 .3 .5],...
             'tag','imAx');

revas.gui.UserData.posAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[.3 .5 .7 .5],...
             'tag','posAx');   
         
revas.gui.UserData.peakAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[0 .25 .5 .25],...
             'tag','peakAx'); 
         
revas.gui.UserData.motAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[0.5 .25 .5 .25],...
             'tag','motAx'); 

revas.gui.UserData.blinkAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[0.5 0 .5 .25],...
             'tag','blinkAx'); 
         
revas.gui.UserData.stimAx = axes(revas.gui.UserData.visualizePanel,...
             'units','normalized',...
             'FontSize',revas.fontSize,...
             'outerposition',[0 0 .5 .25],...
             'tag','stimAx'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file selection panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
             'visible','off',...
             'min',1,...
             'max',3);         
         
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pipeline panel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
gParams.enableVerbosity = 'no';
gParams.enableGPU = 0;
gParams.enableParallel = 0;
gParams.saveLog = 1;
gParamsNames = fieldnames(gParams);

% make sure enableVerbosity is at the top
iv = contains(gParamsNames,'enableVerbosity');
gParamsNames = [gParamsNames(iv); gParamsNames(~iv)];

rowSize = 0.8 / (length(gParamsNames)+1);

for i=1:length(gParamsNames)

    % location of the current uicontrol
    yLoc = 0.85 - (i)*rowSize - (i>1)*0.1;
    xLoc = 0.1;
    thisPosition = [xLoc yLoc 0.75 rowSize];
    
    % disable fields if there is no GPU or multiple logical cores
    enable = IsEnabledUI(gParamsNames{i});
    
    % create the uicontrol. create a checkbox for all options except for
    % enableVerbosity, which needs three options, therefore we create a
    % popup menu for it.
    if strcmp(gParamsNames{i}, 'enableVerbosity')
        revas.gui.UserData.((gParamsNames{i})) = ...
            uicontrol(revas.gui.UserData.globalPanel,...
            'style','popup',...
            'units','normalized',...
            'position',thisPosition,...
            'fontsize',revas.fontSize,...
            'string',{'no','yes','module'},...
            'value',1,...
            'callback',{@GParamsCallback,revas},...
            'tag',gParamsNames{i},...
            'enable',enable);
        
        % create the label
        revas.gui.UserData.([gParamsNames{i} 'Label']) = ...
            uicontrol(revas.gui.UserData.globalPanel,...
            'style','text',...
            'units','normalized',...
            'position',[thisPosition(1) thisPosition(2)+rowSize thisPosition(3:4)],...
            'fontsize',revas.fontSize,...
            'horizontalalignment','center',...
            'string','Visualize results?');
    else
        revas.gui.UserData.((gParamsNames{i})) = ...
            uicontrol(revas.gui.UserData.globalPanel,...
            'style','checkbox',...
            'units','normalized',...
            'position',thisPosition,...
            'fontsize',revas.fontSize,...
            'string',gParamsNames{i},...
            'value',double(gParams.(gParamsNames{i})),...
            'callback',{@GParamsCallback},...
            'tag',gParamsNames{i},...
            'enable',enable);
    end
end
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log, status, and run 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
revas.gui.UserData.logBox = uicontrol(revas.gui,...
             'style','listbox',...
             'units','normalized',...
             'position',logBoxPos,...
             'fontsize',revas.fontSize,...
             'string',{},...
             'value',1,...
             'min',1,...
             'max',3,...
             'callback',{@ShowLogCallback,revas},...
             'tooltip','Log window.');  
         
revas.gui.UserData.runButton = uicontrol(revas.gui,...
             'style','togglebutton',...
             'units','normalized',...
             'position',runPos,...
             'fontsize',revas.fontSize,...
             'string','Run',...
             'value',0,...
             'callback',{@RunPipeline}); 
                
revas.gui.UserData.abortButton = uicontrol(revas.gui,...
             'style','togglebutton',...
             'units','normalized',...
             'position',abortPos,...
             'fontsize',revas.fontSize,...
             'string','Abort',...
             'value',0,...
             'visible','off');          

revas.gui.UserData.statusBar = axes(revas.gui,...
             'units','normalized',...
             'position',statusPos,...
             'xtick',[],'ytick',[],'tag','statusBar');         

revas.gui.UserData.pipeline = {};
revas.gui.UserData.pipeParams = {};
revas.gui.UserData.fileList = {};
revas.gui.UserData.enableState = {};
         
% Make visible 
set(revas.gui,'visible','on');
RevasMessage(sprintf('ReVAS %s launched!',versionNo),revas.gui.UserData.logBox);



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %
    % pipeline menu callbacks
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function RevasReportIssue(varargin)
        fprintf('\n\n%s: Launching web browser for filing an issue.\n',datestr(datetime))
        web('https://github.com/lowvisionresearch/ReVAS/issues', '-browser');
    end
    
    function RevasDocumentation(varargin)
        fprintf('\n\n%s: Launching web browser for ReVAS wiki.\n',datestr(datetime))
        web('https://github.com/lowvisionresearch/ReVAS/wiki', '-browser');
    end

    function enable = IsEnabledUI(paramName)
        enable = true;
        switch paramName
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
    end

    function RunMenuCallback(varargin)
        isSelected = varargin{3};
        
        % run with all files
        if ~isSelected
            % select all files
            revas.gui.UserData.lbFile.Value = 1:length(revas.gui.UserData.lbFile.String);
        end
        
        % simulate button press on Run
        revas.gui.UserData.runButton.Value = 1;
        
        % call RunPipeline
        RunPipeline(revas.gui.UserData.runButton);
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
                revas.gui.UserData.isChange = CompareFieldsHelper(oldParams,newParams);
                if revas.gui.UserData.isChange
                    revas.gui.UserData.pipeParams{val,1} = newParams;
                    str = evalc('disp(newParams)');
                    RevasMessage(sprintf('New parameters for %s: \n %s',thisModule{1},str),revas.gui.UserData.logBox)
                end
            end
        end
        
        % if there is any change, enable save menus
        if revas.gui.UserData.isChange
            % enable save and saveas menus
            % get siblings 
            childrenObjs = get(revas.pipelineMenu,'children');
            set(childrenObjs(contains({childrenObjs.Text},{'Save','Save As'})),'Enable','on');
        end
    end


    function GParamsCallback(src,~,varargin)
        
        if strcmp(src.Tag,'enableVerbosity')
            gParams.(src.Tag) = src.String{src.Value};
        else
            gParams.(src.Tag) = src.Value;
        end
        
        switch src.Tag
            case 'enableGPU'
                gParams.enableParallel = ~src.Value & feature('numcores') > 1 & gParams.enableParallel;
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
        
        str = [src.Tag ': ' num2str(src.Value)];
        RevasMessage(sprintf('Global Flag changed: %s',str),revas.gui.UserData.logBox);
        RefreshGlobalFlagPanel;
    end

    function RefreshGlobalFlagPanel
        for fld=1:length(gParamsNames)
            thisField = gParamsNames{fld};
            if strcmp(thisField,'enableVerbosity')
                revas.gui.UserData.(thisField).Value = find(contains(revas.gui.UserData.(thisField).String,gParams.(thisField)));
            else
            	revas.gui.UserData.(thisField).Value = double(gParams.(thisField));
            end
        end
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %
    %  Run Pipeline
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function RunPipeline(src,varargin)
        
        if ~src.Value
            return;
        end
        
        % Check if we have at least one file in file list 
        if isempty(revas.gui.UserData.fileList)
            warndlg('Please use File Menu to select some files.',...
                'No file selected.','modal');
            src.Value = 0;
            return;
        else
            % check if we have inmemory files before we dive into the
            % pipeline. If so, override the inmemory setting on GUI and set
            % it to true.
            if any(contains(revas.gui.UserData.fileList,'_inmemory.mat'))
                gParams.inMemory = 1;
                revas.gui.UserData.inMemory.Value = 1;
                RevasWarning('inMemory option must be enabled when using _inmemory.mat files. ReVAS is switching InMemory flag on.',revas.gui.UserData.logBox);
            end
        end
        
        % Check if we have at least and one module in the pipeline
        if isempty(revas.gui.UserData.pipeline)
            warndlg('Please use Pipeline Menu to open an existing pipeline or create a new one.',...
                'No pipeline.','modal');
            src.Value = 0;
            return;
        end
        
        
        % all uicontrols that have Enable property
        objs = findall([findobj(revas.gui,'type','uipanel'); ...
                        findobj(revas.gui,'type','uimenu')], '-property', 'Enable');
        % save the enable state of all uicontrols that have Enable property
        revas.gui.UserData.enableState = get(objs,'Enable');
        set(objs, 'Enable', 'off');
        
        % make the abort button visible
        revas.gui.UserData.abortButton.Visible = 'on';
        
        % make run button invisible
        revas.gui.UserData.runButton.Visible = 'off';

        % run the rest of the operation within try catch. if an error
        % occurs, we should be able to re-enable the UIControls.
        try
            selectedFiles = revas.gui.UserData.lbFile.Value;
            RevasMessage(sprintf('Pipeline launched with %d selected files...',length(selectedFiles)),revas.gui.UserData.logBox)
            anyError = ExecutePipeline(revas,gParams);
            
        catch runPipeError
            errStr = getReport(runPipeError,'extended','hyperlinks','off');
            RevasError(sprintf(' \n %s',errStr),revas.gui.UserData.logBox);
        end
        
        if revas.gui.UserData.abortButton.Value
            RevasMessage('Pipeline was aborted by user.',revas.gui.UserData.logBox);
            revas.gui.UserData.abortButton.Value = 0;
        else
            if ~exist('runPipeError','var') && ~anyError
                RevasMessage('All files have been processed. Pipeline stopped.',revas.gui.UserData.logBox);
            else
                RevasMessage('Pipeline stopped with an error. Check the log.',revas.gui.UserData.logBox);
            end
        end
        
        % restore enable state of UIControls
        arrayfun(@(x,y) set(x,'Enable',y{1}), objs, revas.gui.UserData.enableState);
        
        % make abort button hidden again
        revas.gui.UserData.abortButton.Visible = 'off';
        
        % set runButton to default state
        revas.gui.UserData.runButton.Visible = 'on';
        src.Value = 0;
        
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %
    %  Close up 
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function RevasClose(varargin)
        
        % if Run button is on, i.e., a pipeline is running, do not exit.
        % Give a warning only.
        if revas.gui.UserData.runButton.Value
            answer = questdlg('Pipeline is running... Please abort the process and then quit ReVAS.',...
                'Unfinished Business','Exit Anyway','Cancel','modal');
            % Handle response
            switch answer
                case 'Exit Anyway'
                    % do nothing
                case 'Cancel'
                    return;
                otherwise
                    % do nothing
            end
            
        end
        
        % if there are unsaved changes to the pipeline, ask user if she
        % wants to save them.
        if revas.gui.UserData.isChange
            answer = questdlg('There are unsaved changes to the pipeline. What do you want to do?', ...
                'Oops! Unsaved changes to pipeline', ...
                'Exit w/o Saving','Cancel','Save & Exit','Save & Exit');
            % Handle response
            switch answer
                case 'Exit w/o Saving'
                    % do nothing
                case 'Cancel'
                    return;
                case 'Save & Exit'
                    saveMenu = findobj(revas.pipelineMenu,'Text','Save');
                    SavePipeline(saveMenu,[],revas);
                otherwise 
                    % do nothing
            end
        end
        RevasMessage(sprintf('ReVAS closed!\n'),revas.gui.UserData.logBox)
            
        diary off;
        delete(revas.gui);
    end


end




%         % if log file grows too big, rename it and open a new file.
%         if exist(logFile,'file')
%             stats = dir(logFile);
%             if stats.bytes > 10^8
%                 movefile(logFile,...
%                     [logFile(1:end-4) '-' regexprep(datestr(datetime),'[ :]','-') '.txt'])
%             end
%         end