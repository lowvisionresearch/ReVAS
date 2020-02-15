function RevasMessage(message, logBoxHandle)
%REVAS MESSAGE  Issues a message to the GUI's command window text box if possible.
%   Issues a message to the GUI's command window text box if possible.

timeStr = datestr(datetime);
logStr = sprintf('%s: %s',timeStr,message);

% if available, print the message to REVAS GUI
if nargin > 1 && ~isempty(logBoxHandle)
    logBoxHandle.String = [{logStr}; logBoxHandle.String];
end

% get system color for text
persistent textColor
if isempty(textColor)
    c = com.mathworks.services.Prefs.getColorPref('ColorsText');
    textColor = [get(c,'Red') get(c,'Green') get(c,'Blue')]/255;
end

% display the same message via MATLAB command window
cprintf(textColor,'%s\n',logStr);

    

