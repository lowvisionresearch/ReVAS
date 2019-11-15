function SampleRunner
% SampleRunner
%  An example script of how to run the modules.
%
%  MTS 8/22/19 wrote the initial version

%% Example of running pipeline without writing result after each module.

% Ensure abortTriggered is false!
global abortTriggered
abortTriggered = false;

tic;
% inputVideoPath = fullfile(pwd, 'demo/sample10deg.avi');
% outputVideoPath = fullfile(pwd, 'demo/sampleRunnerResult.avi');

inputVideoPath = '/Users/mnagaoglu/Personal/sample tslo video/aoslo/20092L_003.avi';
outputVideoPath = '/Users/mnagaoglu/Personal/sample tslo video/aoslo/20092L_003_result.avi';



% Read the input video and pass the matrix into the functions to skip
% writing intermediate videos to file after each module.
reader = VideoReader(inputVideoPath);
numberOfFrames = reader.Framerate * reader.Duration;

% preallocate
video = zeros(reader.Height, reader.Width, numberOfFrames, 'uint8');

for frameNumber = 1:numberOfFrames
	frame = readFrame(reader);
	[~, ~, numChannels] = size(frame);
	if numChannels == 1
		video(1:end, 1:end, frameNumber) = frame;
	else
		video(1:end, 1:end, frameNumber) = rgb2gray(frame);
	end
end


% Run desired modules.
% Also see example usages in the header comment of each module file.
parametersStructure = struct;
parametersStructure.borderTrimAmount = [0 0 12 0];
video = TrimVideo(video, parametersStructure);

% stimulus = struct;
% stimulus.size = 11;
% stimulus.thickness = 1;
% video = RemoveStimuli(video, stimulus, parametersStructure);

parametersStructure = struct;
parametersStructure.isHistEq = false;
parametersStructure.isGammaCorrect = true;
video = GammaCorrect(video, parametersStructure);

parametersStructure = struct;
video = BandpassFilter(video, parametersStructure);

parametersStructure = struct;
parametersStructure.scalingFactor = 0.5;
parametersStructure.refFrameNumber = 1;
parametersStructure.frameIncrement = 5;
refFrame = CoarseRef(video, parametersStructure);

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.enableVerbosity = false;
parametersStructure.stripHeight = 11;
parametersStructure.stripWidth = 500;
parametersStructure.samplingRate = 540;
parametersStructure.enableGaussianFiltering = false;
parametersStructure.maximumPeakRatio = 0.8;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.adaptiveSearch = false;
parametersStructure.enableSubpixelInterpolation = false;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.createStabilizedVideo = false;
parametersStructure.corrMethod = 'fft';
[refFrame, ~, ~] = FineRef(refFrame, video, parametersStructure);

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.enableVerbosity = false;
parametersStructure.stripHeight = 11;
parametersStructure.stripWidth = 500;
parametersStructure.samplingRate = 540;
parametersStructure.enableGaussianFiltering = false;
parametersStructure.maximumPeakRatio = 0.8;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.adaptiveSearch = false;
parametersStructure.enableSubpixelInterpolation = true;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.createStabilizedVideo = true;
parametersStructure.corrMethod = 'fft';
[~, eyeTraces, timeArray, ~] = StripAnalysis(video, refFrame, parametersStructure);

% eyeTraces = FilterEyePosition([eyeTraces timeArray], parametersStructure);

% eyeTraces = ReReference(eyeTraces, refFrame, 'demo/globalRef.tif', parametersStructure);

% parametersStructure.isAdaptive = true;
% [saccades, drifts] = FindSaccadesAndDrifts([eyeTraces timeArray], parametersStructure);

% % Write the video when finished with desired modules.
% writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
% % some videos are not 30fps, we need to keep the same framerate as
% % the source video.
% writer.FrameRate=reader.Framerate;
% open(writer);
% for frameNumber = 1:numberOfFrames
%    writeVideo(writer, video(1:end, 1:end, frameNumber));
% end

close(writer);
toc;

%% Example of running pipeline with result videos written between each module.
clearvars;
fprintf('\n\n\n\n *********************************************** \n\n\n');

% Ensure abortTriggered is false!
global abortTriggered
abortTriggered = false;

tic;
% inputVideoPath = fullfile(pwd, 'demo/sample10deg.avi');
% inputVideoPath = '/Users/mnagaoglu/Personal/sample tslo video/aoslo/20092L_003.avi';
% inputVideoPath = '/Users/mnagaoglu/Personal/sample tslo video/aoslo/aoslo-short.avi';
inputVideoPath = 'C:\Users\spencer\Desktop\aoslo/aoslo-short_nostim_gamscaled_bandfilt.avi';



% get a copy of the original video path
originalVideoPath = inputVideoPath;

% Run desired modules.
% Also see example usages in the header comment of each module file.

% parametersStructure = struct;
% parametersStructure.overwrite = true;
% parametersStructure.borderTrimAmount = [0 0 12 0];
% TrimVideo(inputVideoPath, parametersStructure);
% inputVideoPath = Filename(inputVideoPath, 'trim');

parametersStructure = struct;
stimulus = struct;
stimulus.size = 11;
stimulus.thickness = 1;
RemoveStimuli(inputVideoPath, stimulus, parametersStructure);
inputVideoPath = Filename(inputVideoPath, 'removestim');

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.isHistEq = false;
parametersStructure.isGammaCorrect = true;
GammaCorrect(inputVideoPath, parametersStructure);
inputVideoPath = Filename(inputVideoPath, 'gamma');

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.smoothing = 0.5;
parametersStructure.lowSpatialFrequencyCutoff = 3;
BandpassFilter(inputVideoPath, parametersStructure);
inputVideoPath = Filename(inputVideoPath, 'bandpass');

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.thresholdValue = 4; 
parametersStructure.singleTail = true;
parametersStructure.upperTail = false;
badFrames = FindBlinkFrames(inputVideoPath, parametersStructure);

% parametersStructure = struct;
% parametersStructure.overwrite = true;
% parametersStructure.enableVerbosity = false;
% parametersStructure.scalingFactor = 1;
% parametersStructure.refFrameNumber = 1;
% parametersStructure.frameIncrement = 3;
% CoarseRef(inputVideoPath, parametersStructure);
% refFramePath = Filename(inputVideoPath, 'coarseref');

% % reader = VideoReader(inputVideoPath);
% % coarseRef = readFrame(reader);
% % 
% % parametersStructure = struct;
% % parametersStructure.overwrite = true;
% % parametersStructure.enableVerbosity = false;
% % parametersStructure.stripHeight = 3;
% % parametersStructure.stripWidth = 500;
% % parametersStructure.samplingRate = 3000;
% % parametersStructure.enableGaussianFiltering = false;
% % parametersStructure.maximumPeakRatio = 0.55;
% % parametersStructure.minimumPeakThreshold = 0;
% % parametersStructure.adaptiveSearch = true;
% % parametersStructure.enableSubpixelInterpolation = false;
% % parametersStructure.createStabilizedVideo = false;
% % parametersStructure.corrMethod = 'mex';
% % fineref = FineRef(coarseRef, inputVideoPath, parametersStructure);
% % 
% % parametersStructure = struct;
% % parametersStructure.overwrite = true;
% % parametersStructure.enableVerbosity = false;
% % parametersStructure.stripHeight = 3;
% % parametersStructure.stripWidth = 500;
% % parametersStructure.samplingRate = 3000;
% % parametersStructure.enableGaussianFiltering = false;
% % parametersStructure.maximumPeakRatio = 0.55;
% % parametersStructure.minimumPeakThreshold = 0;
% % parametersStructure.adaptiveSearch = true;
% % parametersStructure.enableSubpixelInterpolation = false;
% % parametersStructure.createStabilizedVideo = false;
% % parametersStructure.corrMethod = 'mex';
% % FineRef(fineref, inputVideoPath, parametersStructure);
% % refFramePath = Filename(inputVideoPath, 'fineref');

% refFrame = imread('/Users/mnagaoglu/Personal/sample tslo video/aoslo/gk/aoslo-short.jpeg');
% load('/Users/mnagaoglu/Personal/sample tslo video/aoslo/gk/aoslo-short-ref.mat');
load('C:\Users\spencer\Desktop\aoslo/gk/aoslo-short-ref.mat');
refFrame = uint8(referenceimage);

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.enableVerbosity = false;
parametersStructure.badFrames = badFrames;
parametersStructure.trim = [0 0];
parametersStructure.stripHeight = 11;
parametersStructure.stripWidth = 500;
parametersStructure.samplingRate = 540;
parametersStructure.enableGaussianFiltering = false;
parametersStructure.maximumPeakRatio = 0.65;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.searchWindowHeight = 79;
parametersStructure.enableSubpixelInterpolation = false;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.corrMethod = 'mex';
if contains(parametersStructure.corrMethod,'fft')
    parametersStructure.adaptiveSearch = false;
else
    parametersStructure.adaptiveSearch = true;
end
parametersStructure.enableGPU = false;
tic;
[rawEyePositionTraces, usefulEyePositionTraces, timeArray] = ...
    StripAnalysis(inputVideoPath, refFrame, parametersStructure); %#ok<*ASGLU>
toc;
tracesPath = Filename(inputVideoPath, 'usefultraces'); %#ok<*NASGU>

[x,y,timeArray2] = FastStripAnalysis(inputVideoPath,refFrame,'opencv',true,0);
usefulEyePositionTraces2 = [x y];

parametersStructure.createStabilizedVideo = true;
parametersStructure.newStripHeight = 1;
parametersStructure.positions = usefulEyePositionTraces;
parametersStructure.time = timeArray;
StabilizeVideo(originalVideoPath, parametersStructure);

% FilterEyePosition(tracesPath, parametersStructure);
% filteredPath = Filename(tracesPath, 'filtered');
% 
% ReReference(filteredPath, refFramePath, 'demo/globalRef.tif', parametersStructure);
% rerefPath = Filename(filteredPath, 'reref');
% 
% parametersStructure.isAdaptive = true;
% [saccades, drifts] = FindSaccadesAndDrifts(rerefPath, parametersStructure);


toc;
%%

load('C:\Users\spencer\Desktop\aoslo\gk\aoslo-short_gamscaled_scaled_bandfilt_meanrem_540_hz_7900.mat','timeaxis_secs','frameshifts_strips')
figure('units','normalized','outerposition',[.1 .1 .3 .4]); 
plot(timeArray,usefulEyePositionTraces - [0 0],...
    timeArray2,usefulEyePositionTraces2,...
    timeaxis_secs,frameshifts_strips - [32 37]); 
ylim([-100 100])
legend('xReVAS','yReVAS','xMNA','yMNA','xGK','yGK')


end