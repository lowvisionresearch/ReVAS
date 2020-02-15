function OpenPipeline(varargin)
% OpenPipeline(varargin)
%
%   A tool to open an existing pipeline file.
%
% Mehmet N. Agaoglu 1/20/2020 

% first argument is the source
src = varargin{1};

% get siblings 
siblingObjs = get(get(src,'parent'),'children');

% the third argument is the handle from main gui
revas = varargin{3};
RevasMessage(sprintf('OpenPipeline launched.'),revas.gui.UserData.logBox);

% dialog box to select a pipe file
[file,path,~] = uigetfile('*.mat','Select a pipeline file');
if file == 0
    RevasMessage(sprintf('OpenPipeline closed without loading a pipeline.'),revas.gui.UserData.logBox);
    return;
end

% create a matfile object
pipelineFile = fullfile(path,file);
m = load(pipelineFile,'pipeline','pipeParams');

% check if this file has required fields
if ~isfield(m,'pipeline') || ~isfield(m,'pipeParams')
    errordlg('Pipeline file must have ''pipeline'' and ''pipeParams'' fields.',...
        'OpenPipeline Error','modal');
    RevasError(sprintf('OpenPipeline closed due to an error. Selected file does not have required fields.'),revas.gui.UserData.logBox);
    return;
end

% assign output
revas.gui.UserData.pipeline = m.pipeline;
revas.gui.UserData.pipeParams = m.pipeParams;
revas.gui.UserData.pipelineFile = pipelineFile;
revas.gui.UserData.lbPipeline.String = m.pipeline;
revas.gui.UserData.lbPipeline.Value = 1;
revas.gui.UserData.lbPipeline.Visible = 'on';

% enable edit and saveas menus
set(siblingObjs(contains({siblingObjs.Text},'Edit') | ...
                contains({siblingObjs.Text},'Save As')),'Enable','on');
            

RevasMessage(sprintf('Existing pipeline loaded from %s.',pipelineFile),revas.gui.UserData.logBox);

