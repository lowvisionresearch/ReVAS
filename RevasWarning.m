function RevasWarning(message, parametersStructure)
%REVAS WARNING  Issues a warning to the GUI's command window text box if possible.
%   Issues a warning to the GUI's command window text box if possible.

if isfield(parametersStructure, 'commandWindowHandle')
    dateAndTime = datestr(datetime('now'));
    time = dateAndTime(13:20);
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') WARNING: ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
else
    warning(message);
end
    
end