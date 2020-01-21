function SavePipeline(varargin)
% SavePipeline(varargin)
%
%   A tool to overwrite current pipeline file with current pipeline.
%
% Mehmet N. Agaoglu 1/20/2020 

% first argument is the source
src = varargin{1};

% get siblings 
siblingObjs = get(get(src,'parent'),'children');

% the third argument is the handle from main gui
revas = varargin{3};

if ~isfield(revas.gui.UserData,'pipeline') || ...
   ~isfield(revas.gui.UserData,'pipeParams') 
    error('SavePipeline: nothing to save! First load an existing one.')
end

if ~isfield(revas.gui.UserData,'pipelineFile')
    % there is no file name associate with this pipeline, so probably it
    % has been just created. We need Save As to save.
    SaveAsPipeline(src,[],revas);
    return;
end

% overwrite the file with changes
pipeline = revas.gui.UserData.pipeline;
pipeParams = revas.gui.UserData.pipeParams;
save(revas.gui.UserData.pipelineFile,'pipeline','pipeParams');

% disable save menu
set(siblingObjs(contains({siblingObjs.Text},'Save') & ...
               ~contains({siblingObjs.Text},'Save As')),'Enable','off');
