function RevasWarning(message, parametersStructure)
%REVAS WARNING  Issues a warning to the GUI's command window text box if possible.
%   Issues a warning to the GUI's command window text box if possible.

if isfield(parametersStructure, 'commandWindowHandle')
    time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') Warning: ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
else
    warning(message);
end
end
