%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example tests C++ MEX-file detectORBFeaturesOCV and
% extractORBFeaturesOCV. The MEX function detectORBFeaturesOCV uses ORB
% feature detector, and extractORBFeaturesOCV uses ORB feature descriptors
% extractor from OpenCV. These MEX functions are used to automatically
% determine the geometric transformation between a pair of images using
% feature matching. When one image is distorted relative to another by
% rotation and scale, functions |detectORBFeaturesOCV| and
% |estimateGeometricTransform| are used to find the rotation angle and
% scale factor. The distorted image can then be trasformed to recover the
% original image.
%
% Copyright 2014-2017 The MathWorks, Inc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Run the Simulation
% The kernel file
% <matlab:edit('ORBFeaturesCodegen_kernel.m')
% ORBFeaturesCodegen_kernel.m> has two input parameters. The
% first input is the original image and the second input is the image
% distorted by rotation and scale.

% define original image
original = imread('cameraman.tif');
% define distorted image by resizing and then rotating original image
scale = 1.1;
J = imresize(original, scale);
theta = 10;
distorted = imrotate(J, theta);
% call the hand written mex file
disp('% ------- Output from Matlab function call ------');
[matchedOriginalLoc, matchedDistortedLoc,...
    thetaRecovered, ...
  scaleRecovered, recovered] = ...
    ORBFeaturesCodegen_kernel(original, distorted);


%% Run the Generated Code
disp('% ------- Output from Matlab coder generated code ------- ');
[matchedOriginalLocCG, matchedDistortedLocCG,...
   thetaRecoveredCG, scaleRecoveredCG, recoveredCG] = ...
   ORBFeaturesCodegen_kernel_mex(original, distorted);


%% Compare the Generated Code Results with Handwritten MEX Code Results
% Recovered scale and theta for both MEX and generated code, as shown
% above, are within tolerance. Furthermore, the matched point locations are
% identical, as shown below:
disp('% ------- Checking if matched points are identical ------');
isequal(matchedOriginalLocCG, matchedOriginalLoc)
isequal(matchedDistortedLocCG, matchedDistortedLoc)
