function [] = TesterStripAnalysis()
%TESTER FILE FOR STRIP ANALYSIS Run me to test StripAnalysis().
%   Run me to test StripAnalysis().

%% Basic Functionality Test

clc;
clear;
close all;
addpath(genpath('..'));

% videoPath = 'mna_os_10_12_1_45_0_stabfix_17_36_21_409.avi';
videoPath = 'mna_dwt_nostim_nostim_gamscaled_bandfilt_meanrem.avi';
videoFrames = VideoPathToArray(videoPath);
referenceFrame = importdata('ref.mat');
videoWidth = size(videoFrames, 2);

parametersStructure.stripHeight = 15;
parametersStructure.stripWidth = videoWidth;
parametersStructure.samplingRate = 540;
parametersStructure.enableSubpixelInterpolation = true;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.adaptiveSearch = false;
parametersStructure.adaptiveSearchScalingFactor = 8;
parametersStructure.searchWindowHeight = 79;
parametersStructure.badFrames = [29 30];
parametersStructure.minimumPeakRatio = 0.8;
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

tic;

[rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure] ...
    = StripAnalysis(videoPath, referenceFrame, parametersStructure);

toc

fprintf('Process Completed\n');

end

