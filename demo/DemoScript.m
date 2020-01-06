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
inputVideoPath = 'tslo-long.avi'; 
originalVideoPath = inputVideoPath;

% for loading default params, use an empty struct
tp = struct;
tp.overwrite = true;
tp.enableVerbosity = 1;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Find bad frames 
[tp.badFrames, ~, ~, tp.initialRef] = FindBlinkFrames(inputVideoPath, tp);

% Trimming
tp.borderTrimAmount = [0 0 12 12];
[inputVideoPath, tp] = TrimVideo(inputVideoPath, tp);

% Stimulus removal
% tp.stimulus = imread('cross.png');
tp.stimulus = MakeStimulusCross(87, 19, 0); 
inputVideoPath = RemoveStimuli(inputVideoPath, tp);

% Contrast enhancement
inputVideoPath = GammaCorrect(inputVideoPath, tp);

% Bandpass filtering
inputVideoPath = BandpassFilter(inputVideoPath, tp);

% % Make a coarse reference frame using whole-frame template matching
% tp.coarseRef = CoarseRef(inputVideoPath, tp);
% 
% % Make a finer reference frame using strip analysis
% tp.fineRef = FineRef(inputVideoPath, tp);

%%
clc;
tp.adaptiveSearch = true;
tp.axesHandles = [];
tp.referenceFrame = 1;
tp.enableVerbosity = 1;
tp.goodFrameCriterion = 0.5;
tp.swapFrameCriterion = 0.5;
tp.lookBackTime = 15;
tp.trim = tp.borderTrimAmount(3:4);
samplingRate = [540 960];
stripHeight = [15 11];
for i=1:length(stripHeight)
    % Extract eye motion
    tp.axesHandles = [];
    tp.minPeakThreshold = 0.4;
    tp.maxMotionThreshold = 0.3;
    tp.samplingRate = samplingRate(i);
    tp.stripHeight = stripHeight(i);
    tp.stripWidth = [256];
    tp.enableReferenceFrameUpdate = true;
    [position, timeSec, ~, peakValueArray, tp] = StripAnalysis(inputVideoPath, tp); 

    
    % Filter eye motion traces
    positionDeg = -position * 5/512;
    save(tp.outputFilePath,'-append','positionDeg');
    tp.axesHandles = [];
    tp.filters = {'medfilt1','sgolayfilt'};
    tp.filterParams = {15,[3 15]};
    [filteredTraces, timeSec, tp] = FilterEyePosition(tp.outputFilePath, tp);
    
    % Make reference
    tp.axesHandles = [];
    tp.oldStripHeight = tp.stripHeight;
    tp.newStripHeight = tp.stripHeight;
    tp.positions = filteredTraces * -512/5;
    tp.timeSec = timeSec;
    tp.peakValues = peakValueArray;
    tp.maxMotionThreshold = 0.1;
    tp.minPeakThreshold = 0.35;
    [referenceFrame, ~, tp] = MakeReference(inputVideoPath, tp);
    tp.referenceFrame = referenceFrame;
end

% Eye movement classification
tp.axesHandles = [];
[saccades, drifts, labels, tp] =  FindSaccadesAndDrifts([filteredTraces timeSec],  tp);

%% Generate a stabilized video (optionally, using original video)
% StabilizeVideo(originalVideoPath, tp);



%% Re-reference
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
inputVideoPath2 = '20092L_003.avi'; 

% Read the input video into memory
videoArray = ReadVideoToArray(inputVideoPath2);

% for loading default params, use an empty struct
ap = struct;
ap.enableVerbosity = 1;

%%%%%%%%%%%%%%%%%%%%%%%
% Run desired modules.
%%%%%%%%%%%%%%%%%%%%%%%

% Find bad frames 
ap.badFrames = FindBlinkFrames(videoArray, ap);

% Trimming
% videoArray = TrimVideo(videoArray, ap);

% Make a reference frame. 
% ap.refFrame = FineRef([], videoArray, ap);

% Extract eye motion
ap.minPeakThreshold = 0.5;
ap.adaptiveSearch = false;
[position, timeSec] = StripAnalysis(videoArray, ap);

% Post-processing of eye motion traces
position = FilterEyePosition([position, timeSec], ap);

% Classify eye motion into events
[aosloSaccades, aosloDrifts, aosloLabels] = FindSaccadesAndDrifts([position timeSec], ap); %#ok<*ASGLU>

% Visualize results
fh = figure('units','normalized','outerposition',[.1 .1 .5 .4],'name','DemoScript: 2nd example');
PlotResults(fh, position, timeSec, aosloLabels);


