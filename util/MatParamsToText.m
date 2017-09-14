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
