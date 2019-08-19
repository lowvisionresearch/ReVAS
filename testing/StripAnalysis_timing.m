function StripAnalysis_timing
% StripAnalysis_timing
%  Runs the core strip analysis function on the pre-processed demo video,
%  providing timing results for the current implementation.
%
%  MTS 8/19/19 wrote the initial version

addpath(genpath('..'))
clc

parametersStructure = struct;
parametersStructure.overwrite = true;

global abortTriggered;
abortTriggered = 0;

f = @() StripAnalysis( ...
    'demo/sample10deg_dwt_nostim_gamscaled_bandfilt.avi', ...
    'demo/sample10deg_dwt_nostim_gamscaled_bandfilt_refframe.mat', ...
    parametersStructure ...
); % handle to function

warning('off','all')

result = timeit(f, 4);
result

warning('on','all')
end