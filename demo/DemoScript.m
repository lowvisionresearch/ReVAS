%% DemoScript
%  An example script of how to use ReVAS as a toolbox. 
%   
%  There are two examples in this script. The first one takes in takes in a
%  TSLO video and runs through a pipeline while writing out the result of
%  each step. The second example an AOSLO video and goes through another
%  pipeline without writing intermediate results to separate video files.
%  Both examples use default parameters for each video type. The exact
%  values of these parameters were not optimized in a systematic study.
%
%
%  Change history
%  ----------------
%  MTS 8/22/2019  wrote the initial version 
%  MNA 2/21/2020  conformed to new ReVAS after major re-write of core 
%                 pipeline. major clean up, added comments, converted to 
%                 script, added demo videos, loading settings from config 
%                 file.
%
% 


%% First Example 
% Running pipeline with result videos written between each module.

% start fresh
clearvars;
close all;
clc;

fprintf('\n\n\n ------------------- DemoScript 1st Example: TSLO ----------------------- \n\n\n');

% get video path. The demo videos must be under /demo folder, i.e., already
% added to MATLAB path.
inputVideoPath = FindFile('tslo.avi'); 
originalVideoPath = inputVideoPath;

% for loading default params, use an empty struct
tp = struct;
tp.overwrite = true;
tp.enableVerbosity = true;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Trimming
[inputVideoPath, tp] = TrimVideo(inputVideoPath, tp);

% Find bad frames 
[inputVideoPath, tp] = FindBlinkFrames(inputVideoPath, tp);

% Stimulus removal
% tp.stimulus = imread('cross.png'); % example of using an image file
% tp.stimulus = MakeStimulusCross(87, 19, 0); % example of creating a custom cross
inputVideoPath = RemoveStimuli(inputVideoPath, tp);

% Contrast enhancement
[inputVideoPath, tp] = GammaCorrect(inputVideoPath, tp);

% Bandpass filtering
[inputVideoPath, tp] = BandpassFilter(inputVideoPath, tp);

% Strip analysis
[~, tp] = StripAnalysis(inputVideoPath, tp);

% Make reference
[inputVideoPath, tp] = MakeReference(inputVideoPath, tp);

% 2nd round of Strip analysis
[positionPath, tp] = StripAnalysis(inputVideoPath, tp);

% Re-reference
tp.globalRefArgument = imread('tslo-globalRef-tilted-3_25.tif');
tp.fixTorsion = true;
[positionPath, tp] = ReReference(positionPath, tp);

% convert position from pixel to degree
[positionPath, tp] = Pixel2Degree(positionPath, tp);

% eye position filtering
[positionPath, tp] = FilterEyePosition(positionPath, tp);

% eye movement classification
[~, tp] = FindSaccadesAndDrifts(positionPath,  tp);

% Make a new reference using original video and generate a stabilized video
tp.makeStabilizedVideo = true;
tp.newStripHeight = 1;
[~, tp] = MakeReference(originalVideoPath, tp);

% inspect the parameter structure
disp(tp)


%% Second Example 
% Running pipeline completely in memory. 

% start fresh
clearvars;
close all;
clc;

fprintf('\n\n\n ------------------- DemoScript 2nd Example: AOSLO ----------------------- \n\n\n');

% get video path. The demo videos must be under /demo folder, i.e., already
% added to MATLAB path.
inputVideoPath = FindFile('aoslo.avi'); 

% Read the input video into memory
videoArray = ReadVideoToArray(inputVideoPath);

% for loading default params, use an empty struct
ap = struct;
ap.enableVerbosity = true;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Find bad frames 
[videoArray, ap] = FindBlinkFrames(videoArray, ap);

% Stimulus removal
ap.stimulus = imread('cross.png'); % example of using an image file
videoArray = RemoveStimuli(videoArray, ap);

% Contrast enhancement
[videoArray, ap] = GammaCorrect(videoArray, ap);

% Bandpass filtering
[videoArray, ap] = BandpassFilter(videoArray, ap);

% Extract eye motion
ap.enableVerbosity = 'frame';
[~, ap] = StripAnalysis(videoArray, ap);

% convert position from pixel to degree. 
% note that one does not need to use Pixel2Degree module for this. In this 
% example, we will do the conversion ourselves.
ap.fov = 0.83;
ap.frameWidth = 512;
ap.positionDeg = ap.position * ap.fov / ap.frameWidth;

% eye position filtering
[filteredPositionDegAndTime, ap] = FilterEyePosition([ap.positionDeg ap.timeSec], ap);

% Classify eye motion into events
[~,ap] = FindSaccadesAndDrifts(filteredPositionDegAndTime, ap); %#ok<*ASGLU>

% look at the parameters structure
disp(ap)


