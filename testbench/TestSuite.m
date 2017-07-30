function [] = TestSuite()
%TESTER FILE Run me to test ReVAS.
%   Run me to test ReVAS.

%% Video Pre-Processing Test

clc;
clear;
close all;

addpath(genpath('..'));

testVideos = cell(1, 4);
testVideos{1} = 'testbench\mna_os_10_12_1_45_1_stabfix_17_36_21_990.avi';
testVideos{2} = 'testbench\cmo_os_10_4_1_135_1_stabfix_09_33_36_910.avi';
testVideos{3} = 'testbench\djw_os_10_12_1_45_1_stabfix_16_39_42_176.avi';
testVideos{4} = 'testbench\jap_os_10_12_1_45_1_stabfix_11_37_35_135.avi';

benchmarkingVideos = cell(1, 1);
benchmarkingVideos{1} = 'testbench\benchmark\benchmark_realvideos\.avi';

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

parfor i = 1:length(filenames)
    % Grab path out of cell.
    videoPath = filenames{i};
    parametersStructure = struct;
    stimulus = struct;
    
    % Overwrite parameter:
    % if true, recompute and replace existing output file if already present.
    % if false and output file already exists, abort current function/step and continue.
    parametersStructure.overwrite = true;

    % Step 1: Trim the video's upper and right edges.
    parametersStructure.borderTrimAmount = 24;
    TrimVideo(videoPath, parametersStructure);
    fprintf('Process Completed for TrimVideo()\n');
    videoPath = [videoPath(1:end-4) '_dwt' videoPath(end-3:end)]; %#ok<*FXSET>

    % Step 2: Find stimulus location
    parametersStructure.enableVerbosity = false;
    %FindStimulusLocations(videoPath, 'testbench/stimulus_cross.gif', parametersStructure);
    stimulus.thickness = 1;
    stimulus.size = 11;
    FindStimulusLocations(videoPath, stimulus, parametersStructure);
    fprintf('Process Completed for FindStimulusLocations()\n');

    % Step 3: Remove the stimulus
    parametersStructure.overwrite = true;
    RemoveStimuli(videoPath, parametersStructure);
    %copyfile(videoPath, ...
    %    [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]); %#ok<*FXSET>
    fprintf('Process Completed for RemoveStimuli()\n');

    % Step 4: Detect blinks and bad frames
    parametersStructure.thresholdValue = 4;
    FindBlinkFrames(videoPath, parametersStructure);
    fprintf('Process Completed for FindBadFrames()\n');
    % FindBlinkFrames still needs file name from before stim removal.

    % Step 5: Apply gamma correction
    videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]; %#ok<*FXSET>
    parametersStructure.gammaExponent = 0.6;
    GammaCorrect(videoPath, parametersStructure);
    fprintf('Process Completed for GammaCorrect()\n');

    parametersStructure.overwrite = true;
    % Step 6: Apply bandpass filtering
    videoPath = [videoPath(1:end-4) '_gamscaled' videoPath(end-3:end)];
    parametersStructure.smoothing = 1;
    parametersStructure.lowSpatialFrequencyCutoff = 3;
    BandpassFilter(videoPath, parametersStructure);
    fprintf('Process Completed for BandpassFilter()\n');
end

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

%load([videoPath(1:end-4) '_badframes']);
%videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)];
videoFrames = VideoPathToArray(videoPath);
videoWidth = size(videoFrames, 2);

parametersStructure = struct;

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

% First video
videoPath = 'vertical_1_dwt_nostim_gamscaled_bandfilt.avi';
videoFrames = VideoPathToArray(videoPath);
videoWidth = size(videoFrames, 2);
params = struct;
params.videoPath = videoPath;
params.enableSubpixelInterpolation = true;
params.stripHeight = 15;
params.enableGPU = false;
params.samplingRate = 540;
params.adaptiveSearch = false;
params.stripWidth = videoWidth;
params.enableVerbosity = 1;
params.subpixelInterpolationParameters.neighborhoodSize = 7;
params.subpixelInterpolationParameters.subpixelDepth = 2;
params.badFrames = [149 150];
params.enableGaussianFiltering = false; 
params.gaussianStandardDeviation = 10;
params.minimumPeakRatio = 0.8;
params.minimumPeakThreshold = 0;
% params.axesHandles = [];
params.newStripHeight = 11;
params.overwrite = true;
params.numberOfIterations = 1;
params.scalingFactor = 0.4;
coarseRef = CoarseRef(videoPath, params);
load('framePositions.mat')
params.roughEyePositionTraces = framePositions;
FineRef(coarseRef, videoPath, params);

% Second Video
% videoPath = 'djw_os_10_12_1_45_1_stabfix_16_39_42_176_dwt_nostim_gamscaled_bandfilt.avi';
% videoFrames = VideoPathToArray(videoPath);
% videoWidth = size(videoFrames, 2);
% params.videoPath = videoPath;
% params.stripWidth = videoWidth;
% params.fileName = 'djw_os_10_12_1_45_1_stabfix_16_39_42_176_dwt_nostim_gamscaled_bandfilt.avi';
% params.numberOfIterations = 0;
% params.enableGaussianFiltering = false; 
% params.gaussianStandardDeviation = 10;
% coarseRef = CoarseRef(params, 0.4);
% RefineReferenceFrame(coarseRef, params);



end

