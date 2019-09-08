clearvars
close all
clc


% p = gcp('nocreate');
% if isempty(p)
%     p = parpool(4);
% end


% pathToVideo = '../demo/sample10deg';


for i=1:1
    thisFile = 'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi';
    disp('MATLAB normxcorr2 - CPU')
    [x,y] = FastStripAnalysis(thisFile,'matlab',false,1);
    
    disp('MATLAB normxcorr2 - GPU')
    [x,y] = FastStripAnalysis(thisFile,'matlab',true,1);
    
    disp('MATLAB FFT - CPU')
    [x,y] = FastStripAnalysis(thisFile,'fft',false,1);
    
    disp('MATLAB FFT - GPU')
    [x,y] = FastStripAnalysis(thisFile,'fft',true,1);
    
    disp('OPENCV - CPU')
    [x,y] = FastStripAnalysis(thisFile,'opencv',false,1);
    
    disp('OPENCV - GPU')
    [x,y] = FastStripAnalysis(thisFile,'opencv',true,1);
end
legend('matlab normxcorr2 CPU', 'matlab normxcorr2 GPU', 'matlab fft CPU',...
    'matlab fft GPU','opencv CPU','opencv GPU')





