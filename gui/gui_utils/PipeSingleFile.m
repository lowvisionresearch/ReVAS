function PipeSingleFile(inMemory,pipeline,pipeParams,thisFile,fileBox,logBox,fileNo,sPipe,abortButton,overwrite,revas)
% PipeSingleFile(inMemory,pipeline,pipeParams,thisFile,fileBox,logBox,fileNo,sPipe,abortButton,overwrite,revas)
%
%   Running a single file through a pipeline.
%
% MNA 2/2020

[~,shortFileName] = fileparts(thisFile);

% check if pipeline is valid for current file
[tf,startAt,baseFile,skippedIx] = IsValidPipeline(pipeline,thisFile);
if tf && startAt <= length(pipeline)
    RevasMessage(sprintf('Pipeline is valid for %s. Start module is %s.',shortFileName,pipeline{startAt}),logBox);
elseif tf && startAt > length(pipeline)
    RevasWarning(sprintf('Pipeline is valid but this file %s was already analyzed completely with this pipeline.',shortFileName),logBox);
    return;
else
    RevasError(sprintf('Pipeline is invalid for %s. Moving onto next file.',shortFileName),logBox);
    return;
end

% load intermediate .mat file if needed
if ~isempty(skippedIx)
    for i=1:skippedIx
        baseFile = Filename(baseFile,pipeline{i});
    end
    load(baseFile,'params');
end

% to check if this is a video file 
[~,~,ext] = fileparts(thisFile);

% start the pipeline
pipeLength = length(pipeline);
for p = startAt:length(pipeline)
    if abortButton.Value
        break;
    end

    % get a pointer to current module function
    thisModule = str2func(pipeline{p});
    
    % handle first module
    if p==startAt
        
        % handle whether we write out intermediate results or just stay in
        % memory.
        if inMemory
            if strcmpi(ext,'.avi')
                inputArgument = ReadVideoToArray(thisFile);
            else
                d = load(thisFile);
                
                % load params if exists
                if isfield(d,'params')
                    params = d.params;
                end
                
                % load finalOutput if it exist 
                if isfield(d,'finalOutput')
                    inputArgument = d.finalOutput;
                end
                
                % load position and timeSec if exists
                if isfield(d,'position')
                    inputArgument = [d.position d.timeSec];
                end
                
                % load positionDeg and timeSec if exists
                if isfield(d,'positionDeg')
                    inputArgument = [d.positionDeg d.timeSec];
                end
            end
        else
            inputArgument = thisFile;
        end
        
        % get parameters for first module (if not already loaded up to this point)
        if ~exist('params','var')
            params = pipeParams{p};
        end
    end

    % merge cumulative params structure with current module's parameters.
    params = MergeParams(params,pipeParams{p});
    
    % get axes tags
    [~, ~, ~, ~, ~, tags] = GetDefaults(pipeline{p});
    params.axesHandles = [];
    for t=1:length(tags)
        thisAx = revas.gui.UserData.(tags{t});
        
        % to preserve the tag
        set(thisAx,'NextPlot','add');
        
        params.axesHandles = [params.axesHandles; thisAx];
    end
    

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
