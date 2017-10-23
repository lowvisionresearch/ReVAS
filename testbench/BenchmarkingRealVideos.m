function [] = BenchmarkingRealVideos()
%BENCHMARKING Script used to benchmark real videos.
%   Script used to benchmark ReVAS.
%   When prompted, user should select the pre-processed videos.

%% Strip Analysis - Real Videos
% Loads parameters from a mat file and executes coarse, fine, and strip
% modules.

clc;
clear;
close all;
addpath(genpath('..'));

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

coarseParamsCells = cell(1, length(filenames));
fineParamsCells = cell(1, length(filenames));
stripParamsCells = cell(1, length(filenames));

for i = 1:length(filenames)
    originalVideoPath = filenames{i};
    
    nameEnd = strfind(originalVideoPath,'bandfilt');
    paramsPath = [originalVideoPath(1:nameEnd+length('bandfilt')-1) '_params'];
    load(paramsPath, 'coarseParameters', 'fineParameters', 'stripParameters');
    
    % Verbosity flags
    enableVerbosity = false;
    coarseParameters.enableVerbosity = enableVerbosity;
    fineParameters.enableVerbosity = enableVerbosity;
    stripParameters.enableVerbosity = enableVerbosity;

    % Temporary testing
      % FOR TSLO AMD:
%     coarseParameters.searchWindowPercentage = 0.04;
%     fineParameters.searchWindowPercentage = 0.75;
%     stripParameters.searchWindowPercentage = 0.75;
%     
%     coarseParameters.enableGaussianFiltering = 1;
%     coarseParameters.gaussianStandardDeviation = 10;
%     coarseParameters.maximumSD = 30;
%     coarseParameters.SDWindowSize = 25;
%     
%     fineParameters.enableGaussianFiltering = 1;
%     fineParameters.gaussianStandardDeviation = 10;
%     fineParameters.maximumSD = 30;
%     fineParameters.SDWindowSize = 25;
%     
%     stripParameters.enableGaussianFiltering = 1;
%     stripParameters.gaussianStandardDeviation = 10;
%     stripParameters.maximumSD = 30;
%     stripParameters.SDWindowSize = 25;

    coarseParamsCells{i} = coarseParameters; 
    fineParamsCells{i} = fineParameters; 
    stripParamsCells{i} = stripParameters; 
end

parfor i = 1:length(filenames)
    % Grab path out of cell.
    originalVideoPath = filenames{i};

    % MAKE COARSE REFERENCE FRAME
    coarseResult = CoarseRef(originalVideoPath, coarseParamsCells{i});
    fprintf('Process Completed for CoarseRef()\n');

    % MAKE FINE REFERENCE FRAME
    fineResult = FineRef(coarseResult, originalVideoPath, fineParamsCells{i});
    fprintf('Process Completed for FineRef()\n');

    % STRIP ANALYSIS
    StripAnalysis(originalVideoPath, fineResult, fineParamsCells{i});
    fprintf('Process Completed for StripAnalysis()\n');
end
fprintf('Process Completed.\n');

end

