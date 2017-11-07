function [] = BenchmarkingPreProcess()
%BENCHMARKING Script used to pre-process.
%   Script used to benchmark ReVAS.
%   When prompted, user should select raw videos.

%%
clc;
clear;
close all;

addpath(genpath('..'));

CONTAINS_STIM = true;
IS_RODENSTOCK = false;
ONLY_REGENERATE_BLINKS = false;

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
    if ~ONLY_REGENERATE_BLINKS
        if ~IS_RODENSTOCK
            parametersStructure.borderTrimAmount = 24;
            TrimVideo(videoPath, parametersStructure);
        else
            RodenstockTrim(videoPath);
        end
        fprintf('Process Completed for TrimVideo()\n');
    end
    videoPath = [videoPath(1:end-4) '_dwt' videoPath(end-3:end)]; %#ok<*FXSET>

    % Step 2: Find stimulus location
    if CONTAINS_STIM && ~ONLY_REGENERATE_BLINKS
        parametersStructure.enableVerbosity = false; %#ok<UNRCH>
        if ~IS_RODENSTOCK
            %FindStimulusLocations(videoPath, 'testbench/stimulus_cross.gif', parametersStructure);
            stimulus.thickness = 1;
            stimulus.size = 11;
            % For TSLO Normal to remove entire stimulus and black shadow.
            removalAreaSize = [60, 100];
            FindStimulusLocations(videoPath, stimulus, parametersStructure, removalAreaSize);

        else
            stimulus.thickness = 3;
            stimulus.size = 23;
            FindStimulusLocations(videoPath, stimulus, parametersStructure);
        end
        fprintf('Process Completed for FindStimulusLocations()\n');
    end
    
    % Step 3: Remove the stimulus
    if CONTAINS_STIM && ~ONLY_REGENERATE_BLINKS
        RemoveStimuli(videoPath, parametersStructure); %#ok<UNRCH>
        fprintf('Process Completed for RemoveStimuli()\n');
    else
        copyfile(videoPath, ...
            [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]); %#ok<UNRCH,*FXSET> 
    end

    % Step 4: Apply gamma correction
    videoPath = [videoPath(1:end-4) '_nostim' videoPath(end-3:end)]; %#ok<*FXSET>
    if ~ONLY_REGENERATE_BLINKS
        parametersStructure.gammaExponent = 0.6;
        GammaCorrect(videoPath, parametersStructure);
        fprintf('Process Completed for GammaCorrect()\n');
    end

    % Step 5: Apply bandpass filtering
    videoPath = [videoPath(1:end-4) '_gamscaled' videoPath(end-3:end)];
    if ~ONLY_REGENERATE_BLINKS
        parametersStructure.smoothing = 1;
        parametersStructure.lowSpatialFrequencyCutoff = 3;
        BandpassFilter(videoPath, parametersStructure);
        fprintf('Process Completed for BandpassFilter()\n');
    end
        
    % Step 6: Detect blinks and bad frames
    % Default:
    parametersStructure.thresholdValue = inf;
    parametersStructure.upperTail = false;
    %parametersStructure.removalAreaSize = [60, 100];
    %parametersStructure.stitchCriteria = 10;
    % Use the final bandpass filtered video
    videoPath = [videoPath(1:end-4) '_bandfilt' videoPath(end-3:end)];
    FindBlinkFrames(videoPath, parametersStructure);
    fprintf('Process Completed for FindBadFrames()\n');
    % FindBlinkFrames still needs file name from before stim removal.
    
    % ^ Above parameter settings:
    % Black blink:
    %   - upperTail = false;
    % White blink:
    %   - upperTail = true;
    
    PreviewNoBlinkFrames(videoPath);
    fprintf('Process Completed for PreviewNoBlinkFrames()\n');
    
end
fprintf('Process Completed.\n');
end

