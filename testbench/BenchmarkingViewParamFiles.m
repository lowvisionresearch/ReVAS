function [] = BenchmarkingViewParamFiles()
%BENCHMARKING Script used to view param files used to benchmark real videos.
%   Script used to benchmark ReVAS.
%   ** From the file selector, choose the Bandfilt Videos corresponding to the
%   params you want.
%   ** Run this script using Run Section to see variables after it has run.
%   When prompted, user should select the pre-processed videos.

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

clear coarseParameters;
clear filenames;
clear fineParameters;
clear i;
clear nameEnd;
clear originalVideoPath;
clear paramsPath;
clear stripParameters;

fprintf('Process Completed.\n');

end

