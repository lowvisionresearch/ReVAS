function [] = BenchmarkingViewParamFiles()
%BENCHMARKING Script used to view param files used to benchmark real videos.
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

coarseStructs = [];
fineStructs = [];
stripStructs = [];

for i = 1:length(filenames)
    originalVideoPath = filenames{i};
    
    nameEnd = strfind(originalVideoPath,'bandfilt');
    paramsPath = [originalVideoPath(1:nameEnd+length('bandfilt')-1) '_params'];
    load(paramsPath, 'coarseParameters', 'fineParameters', 'stripParameters');
    coarseStructs = [coarseStructs coarseParameters];
    fineStructs = [fineStructs fineParameters];
    stripStructs = [stripStructs stripParameters];
end
struct2table(coarseStructs);
struct2table(fineStructs);
struct2table(stripStructs);
fprintf('Process Completed.\n');

end

