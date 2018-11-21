function RevasError(filename, message, parametersStructure)
%REVAS ERROR  Issues a error message to the GUI's command window text box if possible.
%   Issues a error message to the GUI's command window text box if possible.
%   Execution will continue after the error message is printed if using the GUI.

time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));

% if availabe, print the error message to REVAS GUI
if nargin > 1
    if isfield(parametersStructure, 'commandWindowHandle')

        parametersStructure.commandWindowHandle.String = ...
            ['(' time ') ERROR: ' ...
            'Error while processing ' filename '. Proceeding to next video.) ' ...
            message; ...
            parametersStructure.commandWindowHandle.String];
    end
end

% get system color for text
c = com.mathworks.services.Prefs.getColorPref('Colors_M_Errors');
textColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;

% display the same message via MATLAB command window
cprintf(textColor,'%s\n',(['(' time ') ERROR: '...
    '(Error while processing ' filename '. Proceeding to next video.) '...
    message]));

end
