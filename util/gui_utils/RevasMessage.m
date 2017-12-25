function RevasMessage(message, parametersStructure)
%REVAS MESSAGE  Issues a message to the GUI's command window text box if possible.
%   Issues a message to the GUI's command window text box if possible.

if isfield(parametersStructure, 'commandWindowHandle')
    time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
else
    warning(message);
end    
end
