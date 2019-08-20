%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the kernel for feature matching and
% registration. This is a modified version of the kernel used in
% <matlab:web(fullfile(docroot,'vision','examples','introduction-to-code-generation-with-feature-matching-and-registration.html')); Introduction to Code Generation with Feature Matching and Registration> example.
%
% Copyright 2015-2017 The MathWorks, Inc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [matchedOriginal, matchedDistorted,...
    thetaRecovered, scaleRecovered, recovered] = ...
    ORBFeaturesCodegen_kernel(original, distorted)

%#codegen
coder.extrinsic('featureMatchingVisualization_extrinsic')

%% Step 1: Find Matching Features Between Images
if isMatlabCall()
    ptsOriginal  = detectORBFeaturesOCV(original);
    ptsDistorted = detectORBFeaturesOCV(distorted);
else
    ptsOriginal  = build.detectORBBuildable.detectORBFeatures(original);
    ptsDistorted = build.detectORBBuildable.detectORBFeatures(distorted);
end


% Extract feature descriptors.
if isMatlabCall()
    % Call Matlab function for non-code generation mode. In this mode
    % Matlab calls the MEX-files generated from mex command
    [featuresOriginal_uint8,  validPtsOriginal]  = ...
        extractORBFeaturesOCV(original,  ptsOriginal);
    [featuresDistorted_uint8, validPtsDistorted] = ...
        extractORBFeaturesOCV(distorted, ptsDistorted);
else
    % Generate code using Matlab Coder.The generated code integrates OpenCV
    % functions.
    [featuresOriginal_uint8,  validPtsOriginal]  = ...
      build.extractORBBuildable.extractORBFeatures(original,  ptsOriginal);
    [featuresDistorted_uint8, validPtsDistorted] = ...
      build.extractORBBuildable.extractORBFeatures(distorted, ptsDistorted);
end

featuresOriginal = binaryFeatures(featuresOriginal_uint8);
featuresDistorted = binaryFeatures(featuresDistorted_uint8);

% Match features by using their descriptors.
indexPairs = matchFeatures(featuresOriginal, featuresDistorted, 'MatchThreshold', 80);

% Retrieve locations of corresponding points for each image.
% Note that indexing into the object is not supported in code-generation mode.
% Instead, you can directly access the Location property.
matchedOriginal  = validPtsOriginal.Location(indexPairs(:,1),:);
matchedDistorted = validPtsDistorted.Location(indexPairs(:,2),:);

%% Step 2: Estimate Transformation
% Defaults to RANSAC
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');

%% Step 3: Solve for Scale and Angle
Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc);
thetaRecovered = atan2(ss,sc)*180/pi;

%% Step 4: Recover the original image by transforming the distorted image.
outputView = imref2d(size(original));
recovered  = imwarp(distorted,tform,'OutputView',outputView);

%% Step 5: Display results
featureMatchingVisualization_extrinsic(original,distorted, recovered, ...
    inlierOriginal, inlierDistorted, ...
    matchedOriginal, matchedDistorted, ...
    scaleRecovered, thetaRecovered);

%==========================================================================
function flag = isMatlabCall()
% True if running in MATLAB (not generating code)

flag = isempty(coder.target);
