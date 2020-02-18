function isError = ExecutePipeline(revas,gParams)

isError = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get all variables we need
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
selectedFileIndices = revas.gui.UserData.lbFile.Value;
selectedFiles = revas.gui.UserData.fileList(selectedFileIndices);
pipeline = revas.gui.UserData.pipeline;
pipeParams = revas.gui.UserData.pipeParams;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% handle parallel processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if gParams.enableParallel
    myPool = gcp; 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% handle global flags, also add logBox handle 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(pipeParams)
    % overwrite is always a global override..
    pipeParams{i}.overwrite = gParams.overwrite == 1;
    
    % global enableVerbosity can override each module or can let each one
    % do whatever they want to do.
    switch gParams.enableVerbosity
        case 'no'
            pipeParams{i}.enableVerbosity = false;
        case 'yes'
            pipeParams{i}.enableVerbosity = true;
        case 'module'
            % do nothing
    end 
    
    % send handle to logBox and abortButton to each module
    pipeParams{i}.logBox = revas.gui.UserData.logBox;
    pipeParams{i}.abort = revas.gui.UserData.abortButton;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize statusBar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xlim(revas.gui.UserData.statusBar,[0 1]);
ylim(revas.gui.UserData.statusBar,[0 1]);
sFiles = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0 0 .5 .5]',[1 .6 .3],'EdgeColor','none');
sPipe = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0.5 0.5 1 1]',[1 .6 .3],'EdgeColor','none');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run pipeline 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numFiles = length(selectedFiles);

if gParams.enableParallel

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % parallel processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % send files to queue
    for f = 1:numFiles
        fevalRequests(f) = parfeval(myPool,@PipeSingleFile,0,...
            gParams.inMemory,...
            pipeline,...
            pipeParams,...
            selectedFiles{f},...
            revas.gui.UserData.lbFile,...
            revas.gui.UserData.logBox,...
            selectedFileIndices(f),...
            sPipe,...
            revas.gui.UserData.abortButton,...
            gParams.overwrite); %#ok<AGROW>
    end

    % now check results as they become available and update UI as much as
    % we can.
    for f = 1:numFiles    
        try
            completedIdx = fetchNext(fevalRequests);

            % update the log with diary
            diaryText = fevalRequests(completedIdx).Diary;
            logText = flipud(regexp(diaryText,'[\n]','split')');
            revas.gui.UserData.logBox.String = [logText; ...
                                                revas.gui.UserData.logBox.String];
            % also show that in command line
            fprintf('%s',diaryText);
            
            % look for error
            if contains(diaryText,'ERROR')
                isError = true;
            end

        catch pipeErr
            isError = true;
            errStr = getReport(pipeErr,'extended','hyperlinks','off');
            RevasError(sprintf(' \n %s',errStr),revas.gui.UserData.logBox);
        end

        % update statusBar
        sFiles.XData = [0 f/numFiles f/numFiles 0]';
        drawnow; pause(0.05);
    end
    cancel(fevalRequests);
    
    
else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % single-thread
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for f = 1:numFiles
        try
            PipeSingleFile(gParams.inMemory,...
                pipeline,...
                pipeParams,...
                selectedFiles{f},...
                revas.gui.UserData.lbFile,...
                revas.gui.UserData.logBox,...
                selectedFileIndices(f),...
                sPipe,...
                revas.gui.UserData.abortButton,...
                gParams.overwrite); 
            
        catch pipeErr
            isError = true;
            errStr = getReport(pipeErr,'extended','hyperlinks','off');
            RevasError(sprintf(' \n %s',errStr),revas.gui.UserData.logBox);
        end
        
        % update statusBar
        sFiles.XData = [0 f/numFiles f/numFiles 0]';
        drawnow; pause(0.05);
    end
    
end

% restore statusBar
delete(get(revas.gui.UserData.statusBar,'children'));





function PipeSingleFile(inMemory,pipeline,pipeParams,thisFile,fileBox,logBox,fileNo,sPipe,abortButton,overwrite)


pipeLength = length(pipeline);
for p = 1:length(pipeline)
    if abortButton.Value
        break;
    end

    % get a pointer to current module function
    thisModule = str2func(pipeline{p});
    
    % handle first module
    if p==1
        
        % handle whether we write out intermediate results or just stay in
        % memory.
        if inMemory
            inputArgument = ReadVideoToArray(thisFile);
        else
            inputArgument = thisFile;
        end
        
        % get parameters for first module
        params = pipeParams{p};
    end

    % merge cumulative params structure with current module's parameters.
    params = MergeParams(params,pipeParams{p});
    params.axesHandles = [];

    % cover special cases in connections between modules. 
    % TO-DO: better handling of module connections
    switch pipeline{p}
        case 'StripAnalysis'
            processedVideo = inputArgument;
        case 'MakeReference'
            inputArgument = processedVideo;
        otherwise
            % do nothing
    end

    % run the module
    [inputArgument, params] = thisModule(inputArgument,params);

    % log and status update
    RevasMessage(sprintf('%s done for %s.',pipeline{p},fileBox.String{fileNo}),logBox);
    sPipe.XData = [0 p/pipeLength p/pipeLength 0]';
    
end

% if inMemory mode is requested, we writeout whatever we have at the final
% stage of the pipeline to a file with name reflecting all modules +
% inMemory flag.
if inMemory
    
    % create file name
    for i=1:length(pipeline)
        thisFile = Filename(thisFile,pipeline{i},pipeParams{i});
    end
    thisFile = Filename(thisFile,'inmemory');
    
    finalOutput = inputArgument;
    if ~exist(thisFile,'file') || overwrite
        
        % remove unnecessary fields
        params = RemoveFields(params,{'logBox','axesHandles','abort'}); 
        
        save(thisFile,'params','finalOutput');
    end
        
end
