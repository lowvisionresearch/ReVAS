function ExecutePipeline(revas,gParams)

global abortTriggered;

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
myPool = gcp('nocreate'); % If no pool, do not create new one.
if ~gParams.enableParallel && (isempty(myPool) || myPool.NumWorkers > 1)
    delete(myPool);
    myPool = parpool(1);
end
if gParams.enableParallel && (isempty(myPool) || myPool.NumWorkers < 2)
    myPool = parpool;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% handle global flags, also add logBox handle 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(pipeParams)
    pipeParams{i}.overwrite = gParams.overwrite == 1;
    pipeParams{i}.enableVerbosity = gParams.enableVerbosity == 1;
    pipeParams{i}.logBox = revas.gui.UserData.logBox;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize statusBar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xlim(revas.gui.UserData.statusBar,[0 1]);
ylim(revas.gui.UserData.statusBar,[0 1]);
sFiles = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0 0 .5 .5]',[.3 .6 1],'EdgeColor','none');
sPipe = patch(revas.gui.UserData.statusBar,[0 .000001*[1 1] 0]',[0.5 0.5 1 1]',[.3 .6 1],'EdgeColor','none');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run pipeline 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numFiles = length(selectedFiles);
for f = 1:numFiles
    abortTriggered = revas.gui.UserData.abortButton.Value;
    if abortTriggered
        break;
    end
    
    try
        pipeLength = length(pipeline);
        for p = 1:length(pipeline)
            thisModule = str2func(pipeline{p});
            if p==1
                if gParams.inMemory
                    inputArgument = ReadVideoToArray(selectedFiles{f});
                else
                    inputArgument = selectedFiles{f};
                end
                    
                [inputArgument, params] = thisModule(inputArgument,pipeParams{p});
            else
                params = MergeParams(pipeParams{p},params);
                params.axesHandles = [];
                
                % cover special cases in connections between modules
                switch pipeline{p}
                    case 'StripAnalysis'
                        processedVideo = inputArgument;
                    case 'MakeReference'
                        inputArgument = processedVideo;
                    case 'ReReference'
                        inputArgument = params.referenceFrame;
                    otherwise
                        % do nothing
                end
                        
                [inputArgument, params] = thisModule(inputArgument,params);
            end
            fprintf('%s done\n',pipeline{p});
            sPipe.XData = [0 p/pipeLength p/pipeLength 0]';
        end
    catch pipeErr
        errStr = getReport(pipeErr);
        RevasError(sprintf(' \n %s',errStr),revas.gui.UserData.logBox);
    end
    
    % update statusBar
    sFiles.XData = [0 f/numFiles f/numFiles 0]';
    pause(.1)
end


% restore statusBar
delete(get(revas.gui.UserData.statusBar,'children'));

