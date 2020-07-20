/*
 * Function for fast matching in the Fourier domain on GPU with CUDA 
 *
 * To use: initialize with 
 * cuda_match(paddedReference,invLocalVars,stripHeight,stripWidth,firstTime)
 *  called in cuda_prep(paddedReference,stripHeight,stripWidth,firstTime)
 *
 * Then match each strip with
 * [correlationMap,xPeak,yPeak,peakValue,secondPeakValue] = cuda_match(strip, copyMap);
 *  matches strip to latest initialized paddedReference
 *  copyMap (boolean flag) determines if the entire correlationMap is 
 *  copied from GPU. This allows more downstream processing but 
 *  reduces speed.
 *
 * E Alexander Nov 2019
 * based on code developed with Yi Zong, James Fong, and Jay Shenoy
 */


%% 
Requirements:
- CUDA toolkit (must match version returned by matlab command gpuDevice)
- working compiler (check with mexcuda -setup)

%% Compile with:
mexcuda -lcufft cuda_match.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu

%% Code for use in StripAnalysis.m
%% Setup: call once before big loop

cuda_prep(parametersStructure.referenceFrame,parametersStructure.stripHeight,parametersStructure.stripWidth)
params.outsize =  size(params.referenceFrame)+[params.stripHeight-1,params.stripWidth-1];
params.copyMap = true; % note: this slows things down considerably, much faster to just use peaks

%% Within loop:
case 'cuda'
    [corrmap,xloc,yloc,peak,secondPeak] = cuda_match(thisStrip,params.copyMap);
    if params.copyMap
        correlationMap = fftshift(corrmap);
        correlationMap = correlationMap(1:params.outsize(1),1:params.outsize(2));
    end


