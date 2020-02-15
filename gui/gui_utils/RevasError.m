function RevasError(message, logBoxHandle)
%REVAS ERROR  Issues a error message to the GUI's command window text box if possible.
%   Issues a error message to the GUI's command window text box if possible.
%   Execution will continue after the error message is printed if using the GUI.

timeStr = datestr(datetime);
logStr = sprintf('%s: ERROR: %s',timeStr,message);

% if available, print the message to REVAS GUI.
if nargin > 1 && ~isempty(logBoxHandle)
    logBoxHandle.String = [{logStr}; logBoxHandle.String];
end

% get system color for text
persistent errColor
if isempty(errColor)
    c = com.mathworks.services.Prefs.getColorPref('Colors_M_Errors');
    errColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;
end

% display the same message via MATLAB command window
cprintf(errColor,'%s\n',logStr);



