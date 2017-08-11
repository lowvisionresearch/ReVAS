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
    
    % Remember video path to make it more convenient to compare params
    % files.
    coarseParameters.originalVideoPath = originalVideoPath;
    fineParameters.originalVideoPath = originalVideoPath;
    stripParameters.originalVideoPath = originalVideoPath;
    
    % Read the videos
    videoArray = VideoPathToArray(originalVideoPath);
    frameHeight = size(videoArray, 1);
    frameWidth = size(videoArray, 2);
    
    % Strip Width
    fineParameters.stripWidth = frameWidth;
    stripParameters.stripWidth = frameWidth;
    
    % Calculate Sampling Rate based on Strip Height
    stripsPerFrame = floor(frameHeight/fineParameters.stripHeight);
    framesPerSecond = 30;
    fineParameters.samplingRate = stripsPerFrame * framesPerSecond;
    
    stripsPerFrame = floor(frameHeight/stripParameters.stripHeight);
    stripParameters.samplingRate = stripsPerFrame * framesPerSecond;
    
    save(paramsPath, 'coarseParameters', 'fineParameters', 'stripParameters');
    fprintf('%d of %d completed.\n', i, length(filenames));
end
fprintf('Process Completed.\n');

end

