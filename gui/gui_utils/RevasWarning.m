function RevasWarning(message, logBoxHandle)
%REVAS WARNING  Issues a warning to the GUI's command window text box if possible.
%   Issues a warning to the GUI's command window text box if possible.

timeStr = datestr(datetime);
logStr = sprintf('%s: WARNING: %s',timeStr,message);

% if available, print the message to REVAS GUI.
if nargin > 1 && ~isempty(logBoxHandle)
    logBoxHandle.String = [{logStr}; logBoxHandle.String];
end

% get system color for text
persistent warnColor
if isempty(warnColor)
    c = com.mathworks.services.Prefs.getColorPref('Colors_M_Warnings');
    warnColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;
end

% display the same message via MATLAB command window
cprintf(warnColor,'%s\n',logStr);



