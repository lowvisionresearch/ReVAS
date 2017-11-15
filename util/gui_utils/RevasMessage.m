function RevasMessage(message, parametersStructure)
%REVAS MESSAGE  Issues a message to the GUI's command window text box if possible.
%   Issues a message to the GUI's command window text box if possible.

if isfield(parametersStructure, 'commandWindowHandle')
    dateAndTime = datestr(datetime('now'));
    time = dateAndTime(13:20);
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
else
    warning(message);
end    
end
