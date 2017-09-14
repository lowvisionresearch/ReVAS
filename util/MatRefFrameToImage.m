function MatRefFrameToImage()
%MAT REF FRAME TO IMAGE Generates image files from ref frame mat files
%   Generates image files from ref frame mat files. We assume the user will
%   select refframe mat files, which contain the reference frame in a variable
%   called refFrame. This script takes that refFrame and re-saves as a jpg
%   with the same name in the same directory. This is useful for
%   generating images for reporting purposes.

%%
addpath(genpath('..'));

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

for i = 1:length(filenames)
    % Grab path out of cell.
    matFilePath = filenames{i};
    jpgFilePath = [matFilePath(1:end-4) '.jpg'];
    
    load(matFilePath);
    imwrite(refFrame,jpgFilePath);
end

fprintf('Process Completed.\n');

end
