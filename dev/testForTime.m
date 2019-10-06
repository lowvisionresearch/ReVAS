clearvars
close all
clc

% TSLO
refFrame = []; %#ok<*NASGU>
thisFile = 'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi';

% AOSLO
% load reference frame
load('/Users/mnagaoglu/Personal/sample tslo video/aoslo/gk/aoslo-short-ref.mat');
refFrame = uint8(referenceimage);

% get video path
thisFile = '/Users/mnagaoglu/Personal/sample tslo video/aoslo/aoslo-short_nostim_gamscaled_bandfilt.avi';

%%
disp('MATLAB normxcorr2 - CPU')
[x1,y1] = FastStripAnalysis(thisFile,refFrame,'matlab',false,1); %#ok<*ASGLU>

%%
disp('MATLAB normxcorr2 - GPU')
[x2,y2] = FastStripAnalysis(thisFile,refFrame,'matlab',true,1);

%%
disp('MATLAB FFT - CPU')
[x3,y3] = FastStripAnalysis(thisFile,refFrame,'fft',false,1);

%%
disp('MATLAB FFT - GPU')
[x4,y4] = FastStripAnalysis(thisFile,refFrame,'fft',true,1);

%%
disp('OPENCV - CPU')
[x5,y5] = FastStripAnalysis(thisFile,refFrame,'opencv',false,1);

%%
disp('OPENCV - GPU')
[x6,y6] = FastStripAnalysis(thisFile,refFrame,'opencv',true,1);

%%
legend('matlab normxcorr2 CPU', 'matlab normxcorr2 GPU', 'matlab fft CPU',...
    'matlab fft GPU','opencv CPU','opencv GPU')





