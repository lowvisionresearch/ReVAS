% Ultimate script to test all combinations of various algorithms to speed
% up strip analysis procedure. Run this one a PC with NVidia GPU.

clearvars
close all
clc

% StripAnalysis.m parameters
p = struct;
p.overwrite = true;
p.enableVerbosity = true;
p.badFrames = [];
p.trim = [0 0];
p.stripHeight = 11;
p.stripWidth = 500;
p.samplingRate = 540;
p.enableGaussianFiltering = false;
p.maximumPeakRatio = 0.65;
p.minimumPeakThreshold = 0;
p.adaptiveSearch = false;
p.searchWindowHeight = 79;
p.enableSubpixelInterpolation = false;
p.subpixelInterpolationParameters.neighborhoodSize = 7;
p.subpixelInterpolationParameters.subpixelDepth = 2;
p.enableGPU = false;

% % TSLO
% refFrame = []; %#ok<*NASGU>
% thisFile = 'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi';

% AOSLO
% load reference frame
load('../aoslo/gk/aoslo-short-ref.mat');
refFrame = uint8(referenceimage);

% get video path
thisFile = '../aoslo/aoslo-short_nostim_gamscaled_bandfilt.avi';

% just to get some stats
videoObj = VideoReader(thisFile);
duration = videoObj.Duration;
frameRate = videoObj.FrameRate;
numOfFrames = frameRate * duration;
numberOfStrips = p.samplingRate * duration;

%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - MATLAB normxcorr2 - Adaptive OFF - GPU OFF')
tic;
p.corrMethod = 'normxcorr';
p.enableGPU = false;
p.adaptiveSearch = false;
n = 1;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);

%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - MATLAB normxcorr2 - Adaptive ON - GPU OFF')
tic;
p.corrMethod = 'normxcorr';
p.enableGPU = false;
p.adaptiveSearch = true;
n = 2;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - MATLAB normxcorr2 - Adaptive OFF - GPU ON')
tic;
p.corrMethod = 'normxcorr';
p.enableGPU = true;
p.adaptiveSearch = false;
n = 3;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - MATLAB normxcorr2 - Adaptive ON - GPU ON ')
tic;
p.corrMethod = 'normxcorr';
p.enableGPU = true;
p.adaptiveSearch = true;
n = 4;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - OPENCV - Adaptive OFF - GPU OFF ')
tic;
p.corrMethod = 'mex';
p.enableGPU = false;
p.adaptiveSearch = false;
n = 5;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - OPENCV - Adaptive ON - GPU OFF ')
tic;
p.corrMethod = 'mex';
p.enableGPU = false;
p.adaptiveSearch = true;
n = 6;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - OPENCV - Adaptive OFF - GPU ON ')
tic;
p.corrMethod = 'mex';
p.enableGPU = true;
p.adaptiveSearch = false;
n = 7;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - OPENCV - Adaptive ON - GPU ON ')
tic;
p.corrMethod = 'mex';
p.enableGPU = true;
p.adaptiveSearch = true;
n = 8;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - FFT - Adaptive OFF - GPU OFF ')
tic;
p.corrMethod = 'fft';
p.enableGPU = false;
p.adaptiveSearch = false;
n = 9;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - FFT - Adaptive ON - GPU OFF ')
tic;
p.corrMethod = 'fft';
p.enableGPU = false;
p.adaptiveSearch = true;
n = 10;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - FFT - Adaptive OFF - GPU ON ')
tic;
p.corrMethod = 'fft';
p.enableGPU = true;
p.adaptiveSearch = false;
n = 11;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('StripAnalysis.m - FFT - Adaptive ON - GPU ON ')
tic;
p.corrMethod = 'fft';
p.enableGPU = true;
p.adaptiveSearch = true;
n = 12;
[raw(:,:,n), ~, tArray(:,n)] = StripAnalysis(thisFile, refFrame, p); %#ok<*ASGLU>
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - MATLAB normxcorr2 - GPU OFF')
tic;
[raw(:,1,13), raw(:,2,13), tArray(:,13)] = FastStripAnalysis(thisFile,refFrame,'matlab',false); %#ok<*ASGLU>
n = 13;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - MATLAB normxcorr2 - GPU ON')
tic;
[raw(:,1,14), raw(:,2,14), tArray(:,14)] = FastStripAnalysis(thisFile,refFrame,'matlab',true);
n = 14;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - FFT - GPU OFF')
tic;
[raw(:,1,15), raw(:,2,15), tArray(:,15)] = FastStripAnalysis(thisFile,refFrame,'fft',false);
n = 15;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - FFT - GPU ON')
tic;
[raw(:,1,16), raw(:,2,16), tArray(:,16)] = FastStripAnalysis(thisFile,refFrame,'fft',true);
n = 16;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);

%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - OPENCV - GPU OFF')
tic;
[raw(:,1,17), raw(:,2,17), tArray(:,17)] = FastStripAnalysis(thisFile,refFrame,'opencv',false);
n = 17;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);


%%
fprintf('--------------------------------------------------\n');
disp('FastStripAnalysis.m - OPENCV - GPU ON')
tic;
[raw(:,1,18), raw(:,2,18), tArray(:,18)] = FastStripAnalysis(thisFile,refFrame,'opencv',true);
n = 18;
timing(n) = toc;
fprintf('Elapsed time: %.5g \t Time per frame: %.5g \t Time per strip: %.5g\n\n\n',timing(n),timing(n)/numOfFrames,timing(n)/numberOfStrips);



keyboard;


%%
clearvars;
clc;
load('timingResults.mat')
labels = cell(18,1);
labels{1} = 'StripAnalysis.m - MATLAB normxcorr2 - Adaptive OFF - GPU OFF';
labels{2} = 'StripAnalysis.m - MATLAB normxcorr2 - Adaptive ON - GPU OFF';
labels{3} = 'StripAnalysis.m - MATLAB normxcorr2 - Adaptive OFF - GPU ON';
labels{4} = 'StripAnalysis.m - MATLAB normxcorr2 - Adaptive ON - GPU ON';
labels{5} = 'StripAnalysis.m - OPENCV - Adaptive OFF - GPU OFF';
labels{6} = 'StripAnalysis.m - OPENCV - Adaptive ON - GPU OFF';
labels{7} = 'StripAnalysis.m - OPENCV - Adaptive OFF - GPU ON';
labels{8} = 'StripAnalysis.m - OPENCV - Adaptive ON - GPU ON';
labels{9} = 'StripAnalysis.m - FFT - Adaptive OFF - GPU OFF';
labels{10} = 'StripAnalysis.m - FFT - Adaptive ON - GPU OFF';
labels{11} = 'StripAnalysis.m - FFT - Adaptive OFF - GPU ON';
labels{12} = 'StripAnalysis.m - FFT - Adaptive ON - GPU ON';
labels{13} = 'FastStripAnalysis.m - MATLAB normxcorr2 - GPU OFF';
labels{14} = 'FastStripAnalysis.m - MATLAB normxcorr2 - GPU ON';
labels{15} = 'FastStripAnalysis.m - FFT - GPU OFF';
labels{16} = 'FastStripAnalysis.m - FFT - GPU ON';
labels{17} = 'FastStripAnalysis.m - OPENCV - GPU OFF';
labels{18} = 'FastStripAnalysis.m - OPENCV - GPU ON';


st.xOffset = squeeze(mode(raw(:,1,1) - raw(:,1,:),1));
st.yOffset = squeeze(mode(raw(:,2,1) - raw(:,2,:),1));
st.timing = timing';
tb = struct2table(st,'RowNames',labels)



