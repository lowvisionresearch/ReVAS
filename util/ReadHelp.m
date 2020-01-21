function helpStr = ReadHelp(callerStr)
% helpStr = ReadHelp(callerStr)
%
% 

filepath = FindFile(callerStr);
text = fileread(filepath);
expression = '(\n{2}?)|(%{2}?)';
matchStr = regexp(text,expression);
helpStr = text(1:matchStr(1));
