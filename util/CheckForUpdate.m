function [isSuccess, msg] = CheckForUpdate

% Check if there is a newer version
currentDir = pwd;
p = which('ReVAS');
cd(fileparts(fileparts(p)));
try
    [~, cmdout] = system('git fetch --dry-run');
catch err0
    msg = sprintf('"git fetch" failed.\n%s',err0.message);
    isSuccess = 0;  
    return;
end

if ~isempty(cmdout)
    % Construct a questdlg with three options
    choice = questdlg(['There is an update version of ReVAS.' ...
        'Would you like to update now?'], ...
        'Update available','Yes','No','Yes');
    % Handle response
    try
        switch choice
            case 'Yes'
                
                !git pull
                msg = 'ReVAS has been updated';
                
            case 'No'
                msg = 'User skipped the update.';
            otherwise 
                % do nothing
        end
    catch err
        msg = sprintf('Update failed!\n%s',err.message);
        isSuccess = 0;  
        return;
    end
    
else
    msg = 'ReVAS is already up to date.';
end

isSuccess = 1;
cd(currentDir);