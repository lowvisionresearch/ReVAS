function [] = TesterStripAnalysis()
%TESTER FILE FOR STRIP ANALYSIS Run me to test StripAnalysis().
%   Run me to test StripAnalysis().

%% Basic Functionality Test

% Input:
% videoInput = path to a 3D array
% referenceFrame = first frame of video
% parametersStructure = as defined below

clc;
clear;
close all;
addpath(genpath('..'));

tic;

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
parametersStructure.searchWindowSize = 0.5;
parametersStructure.adaptiveSearch = false;
parametersStructure.badFrames = 30;
parametersStructure.minimumPeakRatio = 0.8;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.enableVerbosity = false;
parametersStructure.axesHandles = [];
parametersStructure.enableGPU = false;

[rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure] ...
    = StripAnalysis(videoPath, referenceFrame, parametersStructure);

figure(4);
plot(timeArray, usefulEyePositionTraces);
title('Useful Eye Position Traces');
xlabel('Time (sec)');
ylabel('Eye Position Traces (pixels)');
legend('show');
legend('Horizontal Traces', 'Vertical Traces');

toc;

fprintf('Process Completed\n');

end
