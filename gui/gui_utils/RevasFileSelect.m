function RevasFileSelect(varargin)
% RevasFileSelect(varargin)
%
% Callback for File Menu items.
%

% first argument is the source

% the second argument is empty.

% the third argument is the included file types
include = varargin{3};

% the fourth argument is the excldued file types
exclude = varargin{4};

% the fifth argument is the handle to main GUI
revas = varargin{5};
RevasMessage(sprintf('RevasFileSelect launched'),revas.gui.UserData.logBox);

% name of the hidden file that keeps track of last used fileList
fileListFile = revas.gui.UserData.fileListFile;

% if include is an empty array, the callback was triggered by Last Selected
% menu. So we just load the previously used fileList from a hidden file and
% do not initiate the file selection GUI.
if isempty(include)
    load(fileListFile,'fileList')
else
    % select files
    fileList = FileSelect(include, exclude);
    
    % if user cancelled without selection, abort
    if ~iscell(fileList) && fileList == 0
        RevasMessage(sprintf('RevasFileSelect closed without selecting files.'),revas.gui.UserData.logBox);
        return;
    end

    % save the list in a hidden file
    save(fileListFile,'fileList');
    
    % enable the Last Selected menu
    revas.gui.UserData.lastselectedfiles.Enable = 'on';
end

% prepare list for better (less crowded) presentation on GUI
betterList = arrayfun(@(x) regexp(x{1},['(?!.*\' filesep '.*\' filesep ')(.*\..*)'],'match'),fileList);

revas.gui.UserData.fileList = fileList;
revas.gui.UserData.lbFile.String = betterList;
revas.gui.UserData.lbFile.Value = 1;
revas.gui.UserData.lbFile.Visible = 'on';

RevasMessage(sprintf(' %d files have been selected.',length(fileList)),revas.gui.UserData.logBox);





