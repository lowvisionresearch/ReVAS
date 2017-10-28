function [] = BenchmarkingCreateParamFiles()
%BENCHMARKING Script used to create param files to benchmark real videos.
%   Script used to benchmark ReVAS.
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
    
    
    % Blink Params (Can be customized)
    % thresholdvalue was 0.7 for last video in AOSLO
    coarseParameters.thresholdValue = inf;
    coarseParameters.upperTail = true;
    %coarseParameters.removalAreaSize = [60, 100];
    
    % Customized params (for TSLO AMD)
    coarseParameters.minimumPeakThreshold = 0.25;
    coarseParameters.maximumPeakRatio = 0.95;
    coarseParameters.searchWindowPercentage = 0.66;
    
    fineParameters.maximumPeakRatio = 0.9;
    fineParameters.minimumPeakThreshold = 0.25;
    fineParameters.searchWindowPercentage = 0.66;
    fineParameters.numberOfIterations = 2;
    
    stripParameters.minimumPeakThreshold = 0.25;
    stripParameters.maximumPeakRatio = 0.9;
    stripParameters.searchWindowPercentage = 0.66;
    % End of TSLO AMD params
    
    % Customized params for TSLO normal
%     coarseParameters.thresholdValue = inf;
%     coarseParameters.minimumPeakThreshold = 0.2;
%     coarseParameters.maximumPeakRatio = 0.95;
%     coarseParameters.searchWindowPercentage = 0.66;
%     
%     fineParameters.numberOfIterations = 2;
%     fineParameters.minimumPeakThreshold = 0.25;
%     fineParameters.maximumPeakRatio = 0.85;
%     fineParameters.searchWindowPercentage = 0.66;
%     
%     stripParameters.minimumPeakValue = 0.25;
%     stripParameters.maximumPeakRatio = 0.85;
%     stripParameters.searchWindowPercentage = 0.66;
 
    % End of TSLO normal params
    
    
    
    % Customized params for Rodenstock AMD
%     coarseParameters.thresholdValue = 1.25;
%     coarseParameters.upperTail = true;
%     coarseParameters.minimumPeakThreshold = 0.2;
%     coarseParameters.maximumPeakRatio = 0.95;
%     coarseParameters.searchWindowPercentage = 0.5;
%     
%     fineParameters.minimumPeakThreshold = 0.25;
%     fineParameters.maximumPeakRatio = 0.8;
%     fineParameters.numberOfIterations = 1;
%     fineParameters.searchWindowPercentage = 0.5;
%     
%     stripParameters.searchWindowPercentage = 0.5;
%     stripParameters.minimumPeakThreshold = 0.3;
%     stripParameters.maximumPeakRatio = 0.8;
    
    % End of Rodenstock AMD params
    
    
    % Customized params for Rodenstock AMD
%     coarseParameters.thresholdValue = inf;
%     coarseParameters.minimumPeakThreshold = 0.2;
%     coarseParameters.maximumPeakRatio = 0.95;
%     coarseParameters.searchWindowPercentage = 0.5;
%     
%     fineParameters.minimumPeakThreshold = 0.25;
%     fineParameters.maximumPeakRatio = 0.8;
%     fineParameters.numberOfIterations = 2;
%     fineParameters.searchWindowPercentage = 0.5;
%     
%     stripParameters.searchWindowPercentage = 0.5;
%     stripParameters.minimumPeakThreshold = 0.3;
%     stripParameters.maximumPeakRatio = 0.8;
    % End of Rodenstock AMD params
    
    
    % Customized params for Rodenstock Normal
%     coarseParameters.thresholdValue = 1;
%     fineParameters.numberOfIterations = 1;
%     
%     coarseParameters.minimumPeakThreshold = 0.2;
%     coarseParameters.maximumPeakRatio = 0.95;
%     coarseParameters.searchWindowPercentage = 0.7;
%     
%     fineParameters.searchWindowPercentage = 0.7;
%     fineParameters.minimumPeakThreshold = 0.3;
%     
%     stripParameters.minimumPeakThreshold = 0.35;
    
    % End of Rodenstock Normal params
    
    % Customized params (for AOSLO)
%     coarseParameters.minimumPeakThreshold = 0.1;
%     coarseParameters.maximumPeakRatio = 0.85;
%     coarseParameters.searchWindowPercentage = 0.33;
%     
%     fineParameters.searchWindowPercentage = 0.33;
%     stripParameters.searchWindowPercentage = 0.33;
%     fineParameters.numberOfIterations = 3;
    % End of AOSLO parameters
    
    % Other customized params (for TSLO)
    %coarseParameters.searchWindowPercentage = 0.4;
%     fineParameters.searchWindowPercentage = 0.33;
%     stripParameters.searchWindowPercentage = 0.33;
%     
%     coarseParameters.enableGaussianFiltering = 1;
%     coarseParameters.gaussianStandardDeviation = 10;
%     coarseParameters.maximumSD = 30;
%     coarseParameters.SDWindowSize = 25;
%     
%     fineParameters.enableGaussianFiltering = 1;
%     fineParameters.numberOfIterations = 3;
%     fineParameters.gaussianStandardDeviation = 10;
%     fineParameters.maximumSD = 30;
%     fineParameters.SDWindowSize = 25;
%     
%     stripParameters.enableGaussianFiltering = 1;
%     stripParameters.gaussianStandardDeviation = 10;
%     stripParameters.maximumSD = 30;
%     stripParameters.SDWindowSize = 25;
    % End of custom params (for TSLO)
    
    save(paramsPath, 'coarseParameters', 'fineParameters', 'stripParameters');
    fprintf('%d of %d completed.\n', i, length(filenames));
end
fprintf('Process Completed.\n');

end

