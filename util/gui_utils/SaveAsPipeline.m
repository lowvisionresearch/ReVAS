function SaveAsPipeline(varargin)
% SaveAsPipeline(varargin)
%
%   A tool to save changes in current pipeline to a file.
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

% dialog box to select a pipe file name
[file,path,~] = uiputfile('*.mat','Enter a pipeline filename');
if file == 0
    return;
else
    revas.gui.UserData.pipelineFile = fullfile(path,file);
end

% overwrite the file with changes
pipeline = revas.gui.UserData.pipeline;
pipeParams = revas.gui.UserData.pipeParams;
save(revas.gui.UserData.pipelineFile,'pipeline','pipeParams');

% disable save menu
set(siblingObjs(contains({siblingObjs.Text},'Save') & ...
               ~contains({siblingObjs.Text},'Save As')),'Enable','off');
