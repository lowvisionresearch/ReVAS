% tester for Rereference
clearvars;
close all;
clc;

% filename = 'C:\Users\spencer\Desktop\ReVAS\testbench\ReReference\mna_os_10_12_1_45_1_stabfix_17_36_21_990_dwt_nostim_gamscaled_bandfilt_540_hz_final.mat';
% localRefFile = 'C:\Users\spencer\Desktop\ReVAS\testbench\ReReference\ReferenceFrame.mat';
% % globalRefFile = 'C:\Users\spencer\Desktop\ReVAS\testbench\ReReference\mna_os_10deg_global.tif';
% globalRefFile = 'C:\Users\spencer\Desktop\ReVAS\testbench\ReReference\mna_ref_10deg.tif';

path = fileparts(pwd);

filename = [path '\example\normal_dew_os_10_12_1_45_1_stabfix_10_18_41_907_dwt_nostim_gamscaled_bandfilt_540_hz_final.mat'];
localRefFile = [path  '\example\normal_dew_os_10_12_1_45_1_stabfix_10_18_41_907_dwt_nostim_gamscaled_bandfilt_coarseref.mat'];
globalRefFile = [path '\example\dew_os_10deg.tif'];

disp('no torsion correction, simplest peak finder')
tic;
params.enableVerbosity = true;
params.overwrite = true;
params.fixTorsion = false;
params.findPeakMethod = 1;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;



disp('no torsion correction, better peak finder')
tic;
params.enableVerbosity = true;
params.overwrite = true;
params.fixTorsion = false;
params.findPeakMethod = 2;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;



disp('torsion correction, simplest peak finder')
tic;
params.enableVerbosity = true;
params.overwrite = true;
params.fixTorsion = true;
params.findPeakMethod = 1;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;


disp('torsion correction, better peak finder')
tic;
params.enableVerbosity = true;
params.overwrite = true;
params.fixTorsion = true;
params.findPeakMethod = 2;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;

