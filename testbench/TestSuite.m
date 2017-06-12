function [] = TestSuite()
%TESTER FILE Run me to test ReVAS.
%   Run me to test ReVAS.

%% Video Pre-Processing Test

clc;
clear;
close all;

addpath(genpath('..'));

video1 = 'testbench/mna_os_10_12_1_45_1_stabfix_17_36_21_990.avi';
video2 = 'testbench/cmo_os_10_4_1_135_1_stabfix_09_33_36_910.avi';
video3 = 'testbench/djw_os_10_12_1_45_1_stabfix_16_39_42_176.avi';
video4 = 'testbench/jap_os_10_12_1_45_1_stabfix_11_37_35_135.avi';

%for videoPath = {video1}
for videoPath = {video1, video2, video3, video4}
    % Grab path out of cell.
    videoPath = videoPath{1};
    
    % Overwrite parameter:
    % if true, recompute and replace existing output file if already present.
    % if false and output file already exists, abort current function/step and continue.
    parametersStructure.overwrite = true;

    % Step 1: Trim the video's upper and right edges.
    parametersStructure.borderTrimAmount = 24;
    TrimVideo(videoPath, parametersStructure);
    fprintf('Process Completed for TrimVideo()\n');

    % Step 2: Find stimulus location
    videoPath = [videoPath(1:end-4) '_dwt' videoPath(end-3:end)]; %#ok<*FXSET>
    parametersStructure.enableVerbosity = true;
    FindStimulusLocations(videoPath, 'testbench/stimulus_cross.gif', parametersStructure);
    %%stimulus.thickness = 1;
    %%stimulus.size = 11;
    %%FindStimulusLocations(videoPath, stimulus, parametersStructure);
    fprintf('Process Completed for FindStimulusLocations()\n');

    % Step 3: Remove the stimulus
    RemoveStimuli(videoPath, parametersStructure);
    fprintf('Process Completed for RemoveStimuli()\n');

    % Step 4: Detect blinks and bad frames
    parametersStructure.thresholdValue = 4;
    FindBadFrames(videoPath, parametersStructure);
    fprintf('Process Completed for FindBadFrames()\n');

    % Step 5: Apply gamma correction
    videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)];
    parametersStructure.gammaExponent = 0.6;
    GammaCorrect(videoPath, parametersStructure);
    fprintf('Process Completed for GammaCorrect()\n');

    % Step 6: Apply bandpass filtering
    videoPath = [videoPath(1:end-4) '_gamscaled' videoPath(end-3:end)];
    parametersStructure.bandpassSigma = 3;
    BandpassFilter(videoPath, parametersStructure);
    fprintf('Process Completed for BandpassFilter()\n');
end

%% Basic Functionality Test of Strip Analysis

clc;
clear;
close all;
addpath(genpath('..'));

%videoPath = 'mna_os_10_12_1_45_0_stabfix_17_36_21_409.avi';
videoPath = 'mna_dwt_nostim_nostim_gamscaled_bandfilt_meanrem.avi';
%load([videoPath(1:end-4) '_badframes']);
%videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)];
videoFrames = VideoPathToArray(videoPath);
referenceFrame = importdata('ref.mat');
videoWidth = size(videoFrames, 2);

parametersStructure.stripHeight = 15;
parametersStructure.stripWidth = videoWidth;
parametersStructure.samplingRate = 540;
parametersStructure.enableSubpixelInterpolation = true;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.adaptiveSearch = true;
parametersStructure.adaptiveSearchScalingFactor = 8;
parametersStructure.searchWindowHeight = 79;
parametersStructure.badFrames = [29 30];
%parametersStructure.badFrames = badFrames;
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

