function [] = TestSuite()
%TESTER FILE Run me to test ReVAS.
%   Run me to test ReVAS.

%% Basic Functionality Test of Strip Analysis

clc;
clear;
close all;
addpath(genpath('..'));

% Video not pre-processed yet...
videoPath = 'testbench/mna_os_10_12_1_45_0_stabfix_17_36_21_409_dwt_nostim_nostim_gamscaled_bandfilt_meanrem.avi';
referenceFramePath = 'testbench/mna_os_10_12_1_45_0_stabfix_17_36_21_409_dwt_nostim_nostim_gamscaled_bandfilt_meanrem_priorrefdata_720hz.mat';

%videoPath = 'testbench/mna_dwt_nostim_nostim_gamscaled_bandfilt_meanrem.avi';
%referenceFramePath = 'ref.mat';

%videoPath = 'testbench/jap_os_10_12_1_45_-1_stabfix_13_09_01_87_dwt_nostim_nostim_gamscaled_bandfilt_meanrem.avi';
%referenceFramePath = ''; % No reference frame available yet...

%videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)];

parametersStructure = struct;

parametersStructure.stripHeight = 15;
parametersStructure.samplingRate = 540;
parametersStructure.enableSubpixelInterpolation = true;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.adaptiveSearch = true;
parametersStructure.adaptiveSearchScalingFactor = 8;
parametersStructure.searchWindowHeight = 79;
parametersStructure.maximumPeakRatio = 0.8;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.enableVerbosity = true;
parametersStructure.axesHandles = []; % TODO
parametersStructure.enableGPU = false;
% Enable Gaussian Filtering:
%   true => use Gaussian Filtering to determine useful peaks
%   false => use max and second max peaks and peak ratio to determine
%   useful peaks
parametersStructure.enableGaussianFiltering = true; 
parametersStructure.gaussianStandardDeviation = 10;

parametersStructure.overwrite = true;

tic;

[rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure] ...
    = StripAnalysis(videoPath, referenceFramePath, parametersStructure);

toc

fprintf('Process Completed\n');

%% Post-processing Test

parametersStructure.thresholdValue = 6;
parametersStructure.secondaryThresholdValue = 2;
parametersStructure.stitchCriteria = 15;
parametersStructure.minAmplitude = 0.1;
parametersStructure.maxDuration = 100;
parametersStructure.detectionMethod = 2;
parametersStructure.hardVelocityThreshold = 35;
parametersStructure.hardSecondaryVelocityThreshold = 35; % TODO what's the default?
parametersStructure.velocityMethod = 2;
parametersStructure.enableVerbosity = true;
parametersStructure.overwrite = true;

eyePositionTracesPath = ...
    'testbench/jap_os_10_12_1_45_-1_stabfix_13_09_01_87_dwt_nostim_nostim_gamscaled_bandfilt_meanrem_540_hz_final_mehmetsolution.mat';
FindSaccadesAndDrifts(eyePositionTracesPath, [512 512], [10 10], ...
    parametersStructure);

fprintf('Process Completed\n');

%% Fine Reference Frame Test

clear;
clc;
close all;

% First video
videoPath = 'wobble_dwt_nostim_gamscaled_bandfilt.avi';
tracesPath = [videoPath(1:end-4) '_coarseframepositions'];

params = struct;
params.videoPath = videoPath;
params.enableSubpixelInterpolation = true;
params.stripHeight = 15;
params.enableGPU = false;
params.samplingRate = 540;
params.adaptiveSearch = true;
params.adaptiveSearchScalingFactor = 8;
params.searchWindowHeight = 79;
params.enableVerbosity = true;
params.subpixelInterpolationParameters.neighborhoodSize = 7;
params.subpixelInterpolationParameters.subpixelDepth = 2;
params.enableGaussianFiltering = true;
params.gaussianStandardDeviation = 50;
params.maximumPeakRatio = 0.9;
params.minimumPeakThreshold = 0.35;
params.newStripHeight = 11;
params.overwrite = true;
params.numberOfIterations = 1;
params.scalingFactor = 0.4;
coarseRef = CoarseRef(videoPath, params);
%load(tracesPath)
%params.roughEyePositionTraces = framePositions;
%FineRef(coarseRef, videoPath, params);

% Second Video
% videoPath = 'djw_os_10_12_1_45_1_stabfix_16_39_42_176_dwt_nostim_gamscaled_bandfilt.avi';
% params.videoPath = videoPath;
% params.fileName = 'djw_os_10_12_1_45_1_stabfix_16_39_42_176_dwt_nostim_gamscaled_bandfilt.avi';
% params.numberOfIterations = 0;
% params.enableGaussianFiltering = false; 
% params.gaussianStandardDeviation = 10;
% coarseRef = CoarseRef(params, 0.4);
% RefineReferenceFrame(coarseRef, params);


end

