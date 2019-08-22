clearvars
close all
% clc


p = gcp('nocreate');
if isempty(p)
    p = parpool(4);
end


pathToVideo = '../demo/sample10deg_nostim_gamscaled_bandfilt';

tic;
parfor i=1:8
%     thisFile = [pathToVideo ' ' num2str(i) '.avi'];
%     FastStripAnalysis(thisFile,false,0);
    a = randn(10000);
end
toc;
