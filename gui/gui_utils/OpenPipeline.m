function OpenPipeline(varargin)
% OpenPipeline(varargin)
%
%   A tool to open an existing pipeline file.
%
% Mehmet N. Agaoglu 1/20/2020 

fprintf('%s: OpenPipeline launched!\n',datestr(datetime));

% first argument is the source
src = varargin{1};

% get siblings 
siblingObjs = get(get(src,'parent'),'children');

% the third argument is the handle from main gui
revas = varargin{3};

% dialog box to select a pipe file
[file,path,~] = uigetfile('*.mat','Select a pipeline file');
if file == 0
    fprintf('%s: OpenPipeline closed without loading a pipeline.\n',datestr(datetime));
    return;
end

% create a matfile object
pipelineFile = fullfile(path,file);
m = load(pipelineFile,'pipeline','pipeParams');

% check if this file has required fields
if ~isfield(m,'pipeline') || ~isfield(m,'pipeParams')
    errordlg('Pipeline file must have ''pipeline'' and ''pipeParams'' fields.',...
        'OpenPipeline Error','modal');
    fprintf('%s: OpenPipeline closed due to an error. Selected file does not have required fields.\n',datestr(datetime));
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
            
            
fprintf('%s: Existing pipeline loaded from %s.\n',datestr(datetime),pipelineFile);
