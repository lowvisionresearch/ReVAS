%% DemoScript
%  An example script of how to use ReVAS as an API/toolbox. 
%   
%  There are two examples in this script. The first one takes in takes in a
%  TSLO video and runs through all steps while writing out the result of
%  each step. The second example an AOSLO video and goes through all
%  (except remove-stimulus, coarse-ref, and re-referencing) steps without
%  writing intermediate results to separate video files.  Both examples use
%  default parameters for each video type. The exact values of these
%  parameters were not optimized in a systematic study.
%
%
%  Change history
%  ----------------
%  MTS 8/22/19  wrote the initial version 
%  MNA 12/9/19 conformed to new ReVAS after major re-write of core 
%               pipeline. major clean up, added comments, converted to 
%               script, added demo videos, loading settings from config 
%               file.
%
% 

% start fresh
clearvars;
close all;
clc;

%% First Example 
% Running pipeline with result videos written between each module.

fprintf('\n\n\n ------------------- DemoScript 1st Example: TSLO ----------------------- \n\n\n');

% get video path. The demo videos must be under /demo folder, i.e., already
% added to MATLAB path.
inputVideoPath = 'tslo.avi'; 
originalVideoPath = inputVideoPath;

% for loading default params, use an empty struct
tp = struct;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Find bad frames 
[tp.badFrames, ~, ~, tp.initialRef] = FindBlinkFrames(inputVideoPath, tp);

% Trimming
inputVideoPath = TrimVideo(inputVideoPath, tp);

% Stimulus removal
tp.stimulus = imread('cross.png');
inputVideoPath = RemoveStimuli(inputVideoPath, tp);

% Contrast enhancement
inputVideoPath = GammaCorrect(inputVideoPath, tp);

% Bandpass filtering
inputVideoPath = BandpassFilter(inputVideoPath, tp);

% Make a coarse reference frame using whole-frame template matching
tp.coarseRef = CoarseRef(inputVideoPath, tp);

% Make a finer reference frame using strip analysis
tp.fineRef = FineRef(inputVideoPath, tp);

% Extract eye motion
[~,~,~,~,tp] = StripAnalysis(inputVideoPath, tp); 

% Generate a stabilized video (optionally, using original video)
StabilizeVideo(originalVideoPath, tp);

% Post-processing of eye motion traces
[filteredTraces, filteredPath] = FilterEyePosition(tp.outputFilePath, tp);

% Re-reference
tp.globalRef = imread('tslo-global-ref.png');
[rerefTraces, rerefPath] = ReReference(filteredPath, tp);

% Eye movement classification
[tsloSaccades, tsloDrifts, tsloLabels] = FindSaccadesAndDrifts(rerefPath, tp);

% Visualize results
fh = figure('units','normalized','outerposition',[.1 .5 .5 .4],'name','DemoScript: 1nd example');
PlotResults(fh, rerefTraces, timeArray, tsloLabels);



%% Second Example 
% Running pipeline without writing result after each module.

fprintf('\n\n\n ------------------- DemoScript 2nd Example: AOSLO ----------------------- \n\n\n');

% get video path. The demo videos must be under /demo folder, i.e., already
% added to MATLAB path.
inputVideoPath2 = 'aoslo.avi'; 

% Read the input video into memory
videoArray = ReadVideoToArray(inputVideoPath2);

% for loading default params, use an empty struct
ap = struct;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Find bad frames 
ap.badFrames = FindBlinkFrames(videoArray, ap);

% Trimming
videoArray = TrimVideo(videoArray, ap);

% Make a reference frame. 
ap.refFrame = FineRef([], videoArray, ap);

% Extract eye motion
[position, timeSec] = StripAnalysis(videoArray, ap);

% Post-processing of eye motion traces
position = FilterEyePosition([position, timeSec], ap);

% Classify eye motion into events
[aosloSaccades, aosloDrifts, aosloLabels] = FindSaccadesAndDrifts([position timeSec], ap); %#ok<*ASGLU>

% Visualize results
fh = figure('units','normalized','outerposition',[.1 .1 .5 .4],'name','DemoScript: 2nd example');
PlotResults(fh, position, timeSec, aosloLabels);


