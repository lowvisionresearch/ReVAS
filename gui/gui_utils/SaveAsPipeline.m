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
RevasMessage(sprintf('SaveAsPipeline launched.'),revas.gui.UserData.logBox);

if ~isfield(revas.gui.UserData,'pipeline') || ...
   ~isfield(revas.gui.UserData,'pipeParams') 
    errordlg('SavePipeline: nothing to save! First load an existing one.',...
        'SaveAsPipeline error','modal')
    RevasError(sprintf('SaveAsPipeline returned with an error: Nothing to save.'),revas.gui.UserData.logBox);
    return;
end

% dialog box to select a pipe file name
[file,path,~] = uiputfile('*.mat','Enter a pipeline filename');
if file == 0
    RevasMessage(sprintf('SaveAsPipeline is returning without saving. User cancelled the operation.'),revas.gui.UserData.logBox);
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

revas.gui.UserData.isChange = false;
RevasMessage(sprintf('Pipeline has been saved in %s',revas.gui.UserData.pipelineFile),revas.gui.UserData.logBox);