function fileList = FileSelect(include, exclude)
% fileList = FileSelect(include, exclude)
%
% A tool to select certain files. include and exclude are cell arrays of
% chars indicating included and excluded strings.
%
%
% MNA 2/2/2020 mnagaoglu@gmail.com


%% form the regular expression

% exclude
regstr = '^';
if ~isempty(exclude)
    regstr = [regstr '(?!.*('];
    for i=1:length(exclude)
        % look for '.' and replace with '\.'
        exclude{i} = regexprep(exclude{i},'[\.]','\\\.');
        
        % append reg expression
        regstr = [regstr exclude{i} '|']; %#ok<*AGROW>
    end
    regstr = [regstr(1:end-1) ').*)'];
end

% include
regstr = [regstr '.*('];
for i=1:length(include)
    % look for '.' and replace with '\.'
    include{i} = regexprep(include{i},'[\.]','\\\.');
    
    % append reg. expression
    regstr = [regstr include{i} '|'];
end
regstr = [regstr(1:end-1) ')$'];

%% call uipickfiles
fileList = uipickfiles('REFilter',regstr);

if isrow(fileList)
    fileList = fileList';
end

