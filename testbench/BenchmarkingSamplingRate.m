function [] = BenchmarkingSamplingRate()
%BENCHMARKING Script used to benchmark ReVAS.
%   Script used to benchmark ReVAS.

%% Strip Analysis - Sampling Rate
% Varying sampling rate and using the largest strips possible but such that
% there are no overlapping strips.
% Range of strip heights are (5px : 2px : 51px)
% Fixing Fine Ref's strip height at 15px.

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

parfor i = 1:length(filenames)
    % Grab path out of cell.
    originalVideoPath = filenames{i};
    
    % MAKE COARSE REFERENCE FRAME
    coarseParameters = struct;
    coarseParameters.refFrameNumber = 15;
    coarseParameters.scalingFactor = 0.5;
    coarseParameters.overwrite = true;
    coarseParameters.enableVerbosity = false;
    coarseParameters.fileName = originalVideoPath;
    coarseParameters.enableSubpixelInterpolation = false;
    coarseParameters.adaptiveSearch = false;
    coarseParameters.badFrames = [];
    coarseParameters.minimumPeakRatio = 0;
    coarseParameters.enableGaussianFiltering = false;
    coarseParameters.minimumPeakThreshold = 0;
    coarseParameters.enableGPU = false;

    coarseResult = CoarseRef(originalVideoPath, coarseParameters);
    fprintf('Process Completed for CoarseRef()\n');

    % MAKE FINE REFERENCE FRAME
    fineParameters = struct;
    fineParameters.enableVerbosity = false;
    fineParameters.overwrite = true;
    fineParameters.numberOfIterations = 1;
    fineParameters.stripHeight = 15;
    fineParameters.stripWidth = 488;
    fineParameters.samplingRate = 540;
    fineParameters.minimumPeakRatio = 0.8;
    fineParameters.minimumPeakThreshold = 0.2;
    fineParameters.adaptiveSearch = false;
    fineParameters.enableSubpixelInterpolation = true;
    fineParameters.subpixelInterpolationParameters.neighborhoodSize = 7;
    fineParameters.subpixelInterpolationParameters.subpixelDepth = 2;
    fineParameters.enableGaussianFiltering = false; % TODO
    fineParameters.badFrames = []; % TODO
    fineParameters.enableGPU = false;

    fineResult = FineRef(coarseResult, originalVideoPath, fineParameters);
    fprintf('Process Completed for FineRef()\n');

    for stripHeight = 5:6:51
        
        frameHeight = 488;
        stripsPerFrame = floor(frameHeight/stripHeight);
        framesPerSecond = 30;
        samplingRate = stripsPerFrame * framesPerSecond;

        currentVideoPath = [originalVideoPath(1:end-4) '_STRIPHEIGHT-' int2str(stripHeight) ...
            '_SAMPLINGRATE-' int2str(samplingRate) ...
            originalVideoPath(end-3:end)];
        copyfile(originalVideoPath, currentVideoPath);
        
        tic;
        
        % STRIP ANALYSIS
        stripParameters = struct;
        stripParameters.overwrite = true;
        stripParameters.enableVerbosity = false;
        stripParameters.stripHeight = stripHeight;
        stripParameters.stripWidth = 488;
        stripParameters.samplingRate = samplingRate;
        stripParameters.enableGaussianFiltering = true;
        stripParameters.gaussianStandardDeviation = 10;
        stripParameters.minimumPeakRatio = 0.8;
        stripParameters.minimumPeakThreshold = 0;
        stripParameters.adaptiveSearch = false;
        stripParameters.enableSubpixelInterpolation = true;
        stripParameters.subpixelInterpolationParameters.neighborhoodSize = 7;
        stripParameters.subpixelInterpolationParameters.subpixelDepth = 2;
        stripParameters.enableGPU = false;
        stripParameters.badFrames = []; % TODO

        StripAnalysis(currentVideoPath, fineResult, stripParameters);
        
        elapsedTime = toc;
        
        fclose(fopen([currentVideoPath(1:end-4) '_TIMEELAPSED-' int2str(elapsedTime) '.txt'],'wt+'));
        delete(currentVideoPath);
        fprintf('Process Completed for StripAnalysis()\n');
    end
end
fprintf('Process Completed.\n');

end

