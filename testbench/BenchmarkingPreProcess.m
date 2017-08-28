function [] = BenchmarkingPreProcess()
%BENCHMARKING Script used to pre-process.
%   Script used to benchmark ReVAS.

%%
clc;
clear;
close all;

addpath(genpath('..'));

CONTAINS_STIM = true;
SKIP_TRIM = true;

filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

for i = 1:length(filenames)
    % Grab path out of cell.
    videoPath = filenames{i};
    parametersStructure = struct;
    stimulus = struct;
    
    % Overwrite parameter:
    % if true, recompute and replace existing output file if already present.
    % if false and output file already exists, abort current function/step and continue.
    parametersStructure.overwrite = true;

    % Step 1: Trim the video's upper and right edges.
    if ~SKIP_TRIM
        parametersStructure.borderTrimAmount = 0;
        TrimVideo(videoPath, parametersStructure);
        fprintf('Process Completed for TrimVideo()\n');
    end
    videoPath = [videoPath(1:end-4) '_dwt' videoPath(end-3:end)]; %#ok<*FXSET>

    % Step 2: Find stimulus location
    if CONTAINS_STIM
        parametersStructure.enableVerbosity = false; %#ok<UNRCH>
        %FindStimulusLocations(videoPath, 'testbench/stimulus_cross.gif', parametersStructure);
        %stimulus.thickness = 1;
        %stimulus.size = 11;
        % For Rodenstock:
        stimulus.thickness = 3;
        stimulus.size = 23;
            FindStimulusLocations(videoPath, stimulus, parametersStructure);
            fprintf('Process Completed for FindStimulusLocations()\n');
    end
    
    % Step 3: Remove the stimulus
    if CONTAINS_STIM
        RemoveStimuli(videoPath, parametersStructure); %#ok<UNRCH>
        fprintf('Process Completed for RemoveStimuli()\n');
    else
        copyfile(videoPath, ...
            [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]); %#ok<UNRCH,*FXSET> 
    end

    % Step 4: Apply gamma correction
    videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]; %#ok<*FXSET>
    parametersStructure.gammaExponent = 0.6;
    GammaCorrect(videoPath, parametersStructure);
    fprintf('Process Completed for GammaCorrect()\n');

    % Step 5: Apply bandpass filtering
    videoPath = [videoPath(1:end-4) '_gamscaled' videoPath(end-3:end)];
    parametersStructure.smoothing = 1;
    parametersStructure.lowSpatialFrequencyCutoff = 3;
    BandpassFilter(videoPath, parametersStructure);
    fprintf('Process Completed for BandpassFilter()\n');
        
    % Step 6: Detect blinks and bad frames
    % Default:
    parametersStructure.thresholdValue = 1;
    %parametersStructure.thresholdValue = 1;
    parametersStructure.singleTail = false;
    parametersStructure.upperTail = false;
    %parametersStructure.stitchCriteria = 6;
    % Use the final bandpass filtered video
    videoPath = [videoPath(1:end-4) '_bandfilt' videoPath(end-3:end)];
    FindBlinkFrames(videoPath, parametersStructure);
    fprintf('Process Completed for FindBadFrames()\n');
    % FindBlinkFrames still needs file name from before stim removal.
    
end
fprintf('Process Completed.\n');
end

