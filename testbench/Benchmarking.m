function [] = Benchmarking()
%BENCHMARKING Script used to benchmark ReVAS.
%   Script used to benchmark ReVAS.

%% Strip Analysis - Strip Height
% Only varying strip height of strip analysis. (5px : 2px : 51px)
% Fixing sampling rate at 540 Hz.
% Fixing Fine Ref's strip height at 15px.

clc;
clear;
close all;
addpath(genpath('..'));

benchmarkingVideos = cell(1, 7);
benchmarkingVideos{1} = 'testbench\benchmark\7_13_2017_11_53_8\horizontal_1_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{2} = 'testbench\benchmark\7_13_2017_11_53_8\horizontal_2_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{3} = 'testbench\benchmark\7_13_2017_11_53_8\jerky_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{4} = 'testbench\benchmark\7_13_2017_11_53_8\static_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{5} = 'testbench\benchmark\7_13_2017_11_53_8\vertical_1_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{6} = 'testbench\benchmark\7_13_2017_11_53_8\vertical_2_dwt_nostim_gamscaled_bandfilt.avi';
benchmarkingVideos{7} = 'testbench\benchmark\7_13_2017_11_53_8\wobble_dwt_nostim_gamscaled_bandfilt.avi';

for i = 7:7
    % Grab path out of cell.
    originalVideoPath = benchmarkingVideos{i};
    
    % MAKE COARSE REFERENCE FRAME
    coarseParameters = struct;
    coarseParameters.refFrameNumber = 15;
    coarseParameters.scalingFactor = 0.5;
    coarseParameters.overwrite = true;
    coarseParameters.enableVerbosity = false;
    coarseParameters.fileName = originalVideoPath;
    coarseParameters.enableGPU = false;

    coarseResult = CoarseRef(originalVideoPath, coarseParameters);
    fprintf('Process Completed for CoarseRef()\n');

    % MAKE FINE REFERENCE FRAME
    fineParameters = struct;
    fineParameters.enableVerbosity = true;
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

        
    for stripHeight = 21:2:21

        currentVideoPath = [originalVideoPath(1:end-4) '_STRIPHEIGHT-' int2str(stripHeight) originalVideoPath(end-3:end)];
        copyfile(originalVideoPath, currentVideoPath);
    %%    
        tic;
        
        % STRIP ANALYSIS
        stripParameters = struct;
        stripParameters.overwrite = true;
        stripParameters.enableVerbosity = true;
        stripParameters.stripHeight = stripHeight;
        stripParameters.stripWidth = 488;
        stripParameters.samplingRate = 540;
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
        fprintf('Process Completed for StripAnalysis()\n');
        
    end
end

end

