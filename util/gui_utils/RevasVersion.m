function versionStr = RevasVersion

% read the current version and create the text to be shown
fileID = fopen(FindFile('ReVAS/README.md'));
tline = fgetl(fileID);

% remove '# ReVAS' title
ix = strfind(lower(tline),lower('ReVAS'));
if isempty(ix)
    error('RevasAbout: Cannot find version info in README.md!')
end
tline(1:(ix+4)) = [];

% remove empty space
tline(strfind(tline,' ')) = []; 

% assign to output
versionStr = tline;