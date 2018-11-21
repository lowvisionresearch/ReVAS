function RevasMessage(message, parametersStructure)
%REVAS MESSAGE  Issues a message to the GUI's command window text box if possible.
%   Issues a message to the GUI's command window text box if possible.

if isfield(parametersStructure, 'commandWindowHandle')
    time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));
    parametersStructure.commandWindowHandle.String = ...
        ['(' time ') ' ...
        message; ...
        parametersStructure.commandWindowHandle.String];
end

% get system color for text
c = com.mathworks.services.Prefs.getColorPref('ColorsText');
textColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;

% display the same message via MATLAB command window
for i=1:length(parametersStructure.commandWindowHandle.String)
    cprintf(textColor,'%s\n',parametersStructure.commandWindowHandle.String{i});
end
    
end
