function [isUpToDate, newestVersion] = IsUpToDate()
%%IS UP TO DATE   Checks the ReVAS repository to see whether a newer
%%version has been released.
%

% Looks at the version number recorded in the README to identify this
% software's current version number.
thisVersion = fileread('README.md');
thisVersion = regexp(thisVersion, '\d+\.\d+.\d+', 'match');
thisVersion = cellstr(thisVersion);
thisVersion = thisVersion{1};

% Looks online to see whether this version is the most up to date.
newestVersion = webread('http://selab.berkeley.edu/revas-version');
newestVersion = regexp(newestVersion, '\d+\.\d+.\d+', 'match');
newestVersion = cellstr(newestVersion);
newestVersion = newestVersion{1};
isUpToDate = strcmp(newestVersion, thisVersion);
end
