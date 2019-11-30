function RevasWarning(message, parametersStructure)
%REVAS WARNING  Issues a warning to the GUI's command window text box if possible.
%   Issues a warning to the GUI's command window text box if possible.

time = strtrim(datestr(datetime('now'), 'HH:MM:SS PM'));

% if available, print the message to REVAS GUI.
if nargin > 1
    if isfield(parametersStructure, 'commandWindowHandle')
        parametersStructure.commandWindowHandle.String = ...
            ['(' time ') Warning: ' ...
            message; ...
            parametersStructure.commandWindowHandle.String];
    end
end

% % get system color for text
% c = com.mathworks.services.Prefs.getColorPref('Colors_M_Warnings');
% textColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;

% display the same message via MATLAB command window
% cprintf(textColor,'%s\n',(['(' time ') WARNING: ' message]));
warning('off','backtrace')
warning('%s\n',(['(' time ') ' message]));
warning('on','backtrace')

end
