% tester for Rereference
clearvars;
close all;
clc;

filename = 'C:\Users\spencer\Desktop\ReVAS\example\normal_dew_os_10_12_1_45_1_stabfix_10_18_41_907_dwt_nostim_gamscaled_bandfilt_540_hz_final.mat';
localRefFile = 'C:\Users\spencer\Desktop\ReVAS\example\normal_dew_os_10_12_1_45_1_stabfix_10_18_41_907_dwt_nostim_gamscaled_bandfilt_coarseref.mat';
globalRefFile = 'C:\Users\spencer\Desktop\ReVAS\example\dew_os_10deg.tif';

disp('no torsion correction, simplest peak finder')
tic;
params.enableVerbosity = 0;
params.overwrite = 1;
params.fixTorsion = 0;
params.findPeakMethod = 1;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;



disp('no torsion correction, better peak finder')
tic;
params.enableVerbosity = 0;
params.overwrite = 1;
params.fixTorsion = 1;
params.findPeakMethod = 2;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;



disp('torsion correction, simplest peak finder')
tic;
params.enableVerbosity = 0;
params.overwrite = 1;
params.fixTorsion = 1;
params.findPeakMethod = 1;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;


disp('torsion correction, better peak finder')
tic;
params.enableVerbosity = 0;
params.overwrite = 1;
params.fixTorsion = 1;
params.findPeakMethod = 2;

[newEyePositionTraces, outputFilePath, newParams] = ...
    ReReference(filename, localRefFile, globalRefFile, ...
    params);
toc;

