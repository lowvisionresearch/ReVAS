clearvars
close all
% clc


% p = gcp('nocreate');
% if isempty(p)
%     p = parpool(4);
% end


% pathToVideo = '../demo/sample10deg';

tic;
for i=1:1
    thisFile = 'sample10deg_nostim_gamscaled_bandfilt.avi';
    [x,y] = FastStripAnalysis(thisFile,false,1);
end
toc;
