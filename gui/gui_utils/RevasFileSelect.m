function RevasFileSelect(varargin)
% RevasFileSelect(varargin)
%
% Callback for File Menu items.
%

% first argument is the source
src = varargin{1};

% the second argument is empty.

% the third argument is the included file types
include = varargin{3};

% the fourth argument is the excldued file types
exclude = varargin{4};

% the fifth argument is the handle to main GUI
revas = varargin{5};

% select files
fileList = FileSelect(include, exclude);


revas.gui.UserData.fileList = fileList;
revas.gui.UserData.lbFile.String = fileList;
revas.gui.UserData.lbFile.Value = 1;
revas.gui.UserData.lbFile.Visible = 'on';