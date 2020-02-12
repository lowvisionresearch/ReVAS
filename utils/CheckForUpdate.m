function [isSuccess, msg] = CheckForUpdate(parametersStructure)

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

if contains(cmdout,'error') 
    msg = sprintf('git fetch failed!\nCheck your git user name and email settings.\n');
    isSuccess = 0;
    
    
elseif ~isempty(cmdout)
        
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
                isSuccess = 1;
            case 'No'
                msg = 'User skipped the update.';
                isSuccess = 0;
            otherwise 
                % do nothing
        end
    catch err
        msg = sprintf('Update failed!\n%s',err.message);
        isSuccess = 0;  
    end
    
else
    msg = 'ReVAS is already up to date.';
    isSuccess = 1;
end

cd(currentDir);

RevasMessage(sprintf('[[ Checking for update ]] \n%s', msg), parametersStructure);