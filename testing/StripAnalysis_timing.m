function StripAnalysis_timing
% StripAnalysis_timing
%  Runs the core strip analysis function on the pre-processed demo video,
%  providing timing results for the current implementation.
%
%  MTS 8/19/19 wrote the initial version

addpath(genpath('..'))
clc


load('/Users/mnagaoglu/Personal/sample tslo video/aoslo/gk/aoslo-short-ref.mat');
refFrame = uint8(referenceimage);

parametersStructure = struct;
parametersStructure.overwrite = true;
parametersStructure.enableVerbosity = false;
parametersStructure.badFrames = [];
parametersStructure.trim = [0 0];
parametersStructure.downSampleFactor = 1;
parametersStructure.stripHeight = 11;
parametersStructure.stripWidth = 500;
parametersStructure.samplingRate = 540;
parametersStructure.enableGaussianFiltering = false;
parametersStructure.maximumPeakRatio = 0.65;
parametersStructure.minimumPeakThreshold = 0;
parametersStructure.adaptiveSearch = true;
parametersStructure.searchWindowHeight = 79;
parametersStructure.enableSubpixelInterpolation = false;
parametersStructure.subpixelInterpolationParameters.neighborhoodSize = 7;
parametersStructure.subpixelInterpolationParameters.subpixelDepth = 2;
parametersStructure.corrMethod = 'mex';

global abortTriggered;
abortTriggered = 0;

f = @() StripAnalysis( ...
    '/Users/mnagaoglu/Personal/sample tslo video/aoslo/aoslo-short_nostim_gamscaled_bandfilt.avi', ...
    refFrame, ...
    parametersStructure ...
); % handle to function

oldWarningState = warning;
warning('off','all')

result = timeit(f, 4);
result

warning(oldWarningState);

end