function MatParamsToText()
%MAT PARAMS TO TEXT Saves mat params files as text files.
%   Saves mat params files as text files. We assume the user will select 
%   param mat files, which contain the data called coarseParameters,
%   fineParameters, and stripParameters. This script takes
%   that data and re-saves as text file with the simpler raw video
%   video name in the same directory. This is useful for reporting
%   purposes.

%%
addpath(genpath('..'));

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...NEWLINE');
        return;
    end
end

% Go through list of selected items and filter
i = 1;
while i <= size(filenames, 2)
    if isdir(filenames{i})
        % Pull out any files contained within a selected folder
        % Save the path
        folder = filenames{i};
        % Delete the path from the list
        filenames(i) = [];
        
        % Append files to our list of files if they match the suffix
        folderFiles = dir(folder);
        for j = i:size(folderFiles, 1)
            if ~strcmp(folderFiles(j).name,'.') && ~strcmp(folderFiles(j).name,'..') && ...
                    isempty(findstr('.txt', folderFiles(j).name))
                filenames = ...
                    [filenames, {fullfile(folderFiles(j).folder, folderFiles(j).name)}];
            end
        end
    else
        % Only increment if we did not delete an item this iteration.
        i = i + 1;
    end
end

for i = 1:length(filenames)
    % Grab path out of cell.
    matFilePath = filenames{i};
    originalNameEnd = strfind(matFilePath, '_dwt');
    textFilePath = [matFilePath(1:originalNameEnd-1) '.txt'];
    
    load(matFilePath);
    text = 'Coarse Parameters:NEWLINE';
    if isfield(coarseParameters, 'originalVideoPath')
        coarseParameters = rmfield(coarseParameters, 'originalVideoPath');
    end
    coarse = [strcat(fieldnames(orderfields(coarseParameters)), ': ') ...
        strcat(cellfun(@num2str, struct2cell(orderfields(coarseParameters)), ...
        'UniformOutput', false), 'NEWLINE')];
    coarse = strjoin(coarse','');
    text = [text coarse 'NEWLINE'];
    text = [text 'Fine Parameters:NEWLINE'];
    fineParameters.subpixelInterpolationParameters_neighborhoodSize = ...
        fineParameters.subpixelInterpolationParameters.neighborhoodSize;
    fineParameters.subpixelInterpolationParameters_subpixelDepth = ...
        fineParameters.subpixelInterpolationParameters.subpixelDepth;
    fineParameters = rmfield(fineParameters, 'subpixelInterpolationParameters');
    if isfield(fineParameters, 'originalVideoPath')
        fineParameters = rmfield(fineParameters, 'originalVideoPath');
    end
    fine = [strcat(fieldnames(orderfields(fineParameters)), ': ') ...
        strcat(cellfun(@num2str, struct2cell(orderfields(fineParameters)), ...
        'UniformOutput', false), 'NEWLINE')];
    fine = strjoin(fine','');
    text = [text fine 'NEWLINE'];
    text = [text 'Strip Parameters:NEWLINE'];
    stripParameters.subpixelInterpolationParameters_neighborhoodSize = ...
        stripParameters.subpixelInterpolationParameters.neighborhoodSize;
    stripParameters.subpixelInterpolationParameters_subpixelDepth = ...
        stripParameters.subpixelInterpolationParameters.subpixelDepth;
    stripParameters = rmfield(stripParameters, 'subpixelInterpolationParameters');
    if isfield(stripParameters, 'originalVideoPath')
        stripParameters = rmfield(stripParameters, 'originalVideoPath');
    end
    strip = [strcat(fieldnames(orderfields(stripParameters)), ': ') ...
        strcat(cellfun(@num2str, struct2cell(orderfields(stripParameters)), ...
        'UniformOutput', false), 'NEWLINE')];
    strip = strjoin(strip','');
    text = [text strip 'NEWLINE'];
    
    fileID = fopen(textFilePath,'w');
    fprintf(fileID,'%s', text);
    fclose(fileID);
end

fprintf('Process Completed.\n');

end
