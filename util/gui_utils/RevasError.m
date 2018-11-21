function RevasError(filename, message, parametersStructure)
%REVAS ERROR  Issues a error message to the GUI's command window text box if possible.
%   Issues a error message to the GUI's command window text box if possible.
%   Execution will continue after the error message is printed if using the GUI.

if isfield(parametersStructure, 'commandWindowHandle')
    time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') ERROR: ' ...
        '(Error while processing ' filename '. Proceeding to next video.) ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
end

% get system color for text
c = com.mathworks.services.Prefs.getColorPref('Colors_M_Errors');
textColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;

% display the same message via MATLAB command window
for i=1:length(parametersStructure.commandWindowHandle.String)
    cprintf(textColor,'%s\n',parametersStructure.commandWindowHandle.String{i});
end

end
