function StripAnalysis_profile
% StripAnalysis_profile
%  Runs the core strip analysis function on the pre-processed demo video,
%  providing a timing profile for the current implementation.
%
%  MTS 8/19/19 wrote the initial version

addpath(genpath('..'))
clc

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.adaptiveSearch = false;

global abortTriggered;
abortTriggered = 0;

oldWarningState = warning;
warning('off','all')
profile on

[rawEyePositionTraces, usefulEyePositionTraces, timeArray, ...
    statisticsStructure] = StripAnalysis( ...
    'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi', ...
    'demo/sample10deg_dwt_nostim_gamscaled_bandfilt_refframe.mat', ...
    parametersStructure ...
);

profile viewer
warning(oldWarningState);

% profsave
profile off
end