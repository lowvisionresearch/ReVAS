function [] = BenchmarkingCreateParamFiles()
%BENCHMARKING Script used to create param files to benchmark real videos.
%   Script used to benchmark ReVAS.

%%
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

load('testbench/template_params', ...
    'coarseParameters', 'fineParameters', 'stripParameters');

for i = 1:length(filenames)
    originalVideoPath = filenames{i};
    
    nameEnd = strfind(originalVideoPath,'bandfilt');
    paramsPath = [originalVideoPath(1:nameEnd+length('bandfilt')-1) '_params'];
    coarseParameters.originalVideoPath = originalVideoPath;
    fineParameters.originalVideoPath = originalVideoPath;
    stripParameters.originalVideoPath = originalVideoPath;
    save(paramsPath, 'coarseParameters', 'fineParameters', 'stripParameters');
end
fprintf('Process Completed.\n');

end

