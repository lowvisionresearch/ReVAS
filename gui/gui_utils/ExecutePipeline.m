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
    RevasMessage('Openning a parallel pool...',revas.gui.UserData.logBox);
    myPool = gcp; 
    RevasMessage(sprintf('Parallel pool is ready. Connected to %d workers.',myPool.NumWorkers),revas.gui.UserData.logBox);
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
RevasMessage('Distributed global flags.',revas.gui.UserData.logBox);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize statusBar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xlim(revas.gui.UserData.statusBar,[0 1]);
ylim(revas.gui.UserData.statusBar,[0 1]);
sFiles = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0 0 .5 .5]',[1 .6 .3],'EdgeColor','none');
sPipe = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0.5 0.5 1 1]',[1 .6 .3],'EdgeColor','none');

% get axes handles
axesHandles = findobj(revas.gui.UserData.visualizePanel,'type','axes');

% to preserve the tags
for i=1:length(axesHandles)
    set(axesHandles(i),'NextPlot','add');
end

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
        % queue this file to be processed by one of the workers
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
            gParams.overwrite,...
            axesHandles); %#ok<AGROW>
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
                gParams.overwrite,...
                axesHandles); 
            
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





