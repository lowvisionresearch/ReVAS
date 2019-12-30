clear all
close all
clc


g = gpuDevice(1);
reset(g);
% diary('log');
vid = VideoReader('sample.avi');
%%
referenceFrame = single(readFrame(vid));
referenceFrame = referenceFrame(:,1:500);
stripHeight = 16;
stripWidth = 200;
startx = 25;
starty = 100;
strip1 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);
%%
%referenceFrame = single(ones(2048,2048));
%referenceFrame(2025:2035,2035:2045) = 250;

% referenceFrame = single(ones(50,70));
% referenceFrame(25:35,35:45) = 250;

referenceFrame = single(readFrame(vid));
% referenceFrame = referenceFrame(1:70,1:50);

stripHeight = 20;
stripWidth = 512;

startx = 1;
starty = 30;
strip1 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);

startx = 1;
starty = 20;
strip2 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);

figure(); imshow(strip2,[]);
%%
[refHeight,refWidth] = size(referenceFrame);
referenceFrame = referenceFrame-mean(mean(referenceFrame));
paddedReference = mean(mean(referenceFrame))*single(ones(size(referenceFrame)+[stripHeight,stripWidth]));
paddedReference(1:refHeight,1:refWidth) = referenceFrame;
%paddedReference = referenceFrame;

scratch = cumsum(paddedReference,1);
scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
scratch = cumsum(scratch,2);
localSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);

scratch = cumsum(paddedReference.^2,1);
scratch = scratch(1+stripHeight:end-1,:)-scratch(1:end-stripHeight-1,:);
scratch = cumsum(scratch,2);
localSquaredSums = scratch(:,1+stripWidth:end-1)-scratch(:,1:end-stripWidth-1);
localVars = sqrt(localSquaredSums-localSums.^2/(stripHeight*stripWidth));
localVars(localVars==0) = max(max(localVars));
paddedVars = mean(mean(localVars))*single(ones(size(referenceFrame)+[stripHeight,stripWidth]));
paddedVars(1:refHeight-1,1:refWidth-1) = localVars;

%%
reset(g);

%%

[correlationMap, xPeak, yPeak, peakValue] = CUDACorrelation(strip1, referenceFrame, true, true, g);

%%
p.corrMethod = 'mex';
p.rowStart = 1;
p.rowEnd = size(referenceFrame,1);
p.adaptiveSearch = false;
p.referenceFrame = uint8(referenceFrame);
p.enableGPU = false;
[cm, xp, yp, pv] = LocateStrip(uint8(strip1),p, struct);


%%
mexcuda -lcufft -lcublas cuda_match.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu
%%
cuda_match(paddedReference,localVars,stripHeight,stripWidth)
%%
cuda_match(strip2,true);
%%
i = 1;
diff = nan*ones(16,2);
 outsize = size(paddedReference);
for startx = 1:10:31
    for starty = 1:10:31
        strip1 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);
        [corrmap0,xloc,yloc,peak0,peak1] = cuda_match(strip1,false);
        [corrmap,xloc,yloc,peak] = cuda_match(strip1,true);
        dvals = sort(corrmap(:),'descend');
        peak0, peak1, dvals(1:2)
        
        correlationMap = fftshift(corrmap);
        correlationMap = correlationMap(1:outsize(1)-1,1:outsize(2)-1);
        correlationMap(xloc,yloc) = peak;
        correlationMap = correlationMap/max(correlationMap(:)); %TODO: fix normalization
        figure(1); cla; mesh(correlationMap);
        [val, ind] = max(correlationMap(:));
        [a0,b0] = ind2sub(size(correlationMap),ind)
        %a0 = a0-43;
        %b0 = b0-76;

        gold = normxcorr2(strip1,referenceFrame);
        figure; imshow(gold,[]);
        [val, ind] = max(gold(:));
        [a,b] = ind2sub(size(gold),ind)
        
        diff(i,1) = a0-a;
        diff(i,2) = b0-b;
        i = i+1;
    end
end
figure; plot(diff);
mean(diff)
%%
mexcuda -lcufft -lcublas cuda_scratch2.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu
%%
i = 1;
clear diff;
xes = 120:1:130;
for startx = xes
   cuda_scratch2(paddedReference,localSums,localVars,stripHeight,stripWidth,64,startx);
      
        [corrmap,xloc,yloc,peak] = cuda_scratch2(strip1,true);     
        shifted = fftshift(corrmap);
        figure; imshow(shifted,[]);
        [val, ind] = max(shifted(:));
        [a0,b0] = ind2sub(size(shifted),ind);
        %a0 = a0-43;
        %b0 = b0-76;

        gold = normxcorr2(strip1,referenceFrame);
        figure; imshow(gold,[]);
        [val, ind] = max(gold(:));
        [a,b] = ind2sub(size(gold),ind)
        
        diff(i,1) = a0-a;
        diff(i,2) = b0-b;
        i = i+1;
end
figure; plot(xes',diff);
mean(diff)

%%
% timing other methods:
%$
g = gpuDevice(1);
vid = VideoReader('sample.avi');
referenceFrame = readFrame(vid);
inputVideo = 'cuda/sample.avi';
parametersStructure.scalingFactor = 1;
parametersStructure.adaptiveSearch = false;
parametersStructure.downSampleFactor = 1;
parametersStructure.badFrames = [];
%%
reset(g);
parametersStructure.corrMethod  = 'cuda';
parametersStructure.enableGPU = false
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

reset(g);
parametersStructure.corrMethod  = 'mex';
parametersStructure.enableGPU  = false
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

reset(g);
parametersStructure.corrMethod  = 'fft';
parametersStructure.enableGPU  = false
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

% currently hangs
% reset(g);
% parametersStructure.corrMethod  = 'mex';
% parametersStructure.enableGPU  = true
% tic
% StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
% toc

reset(g);
parametersStructure.corrMethod  = 'fft';
parametersStructure.enableGPU  = true
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

reset(g);
parametersStructure.corrMethod  = 'normxcorr';
parametersStructure.enableGPU  = true
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

reset(g);
parametersStructure.corrMethod  = 'normxcorr';
parametersStructure.enableGPU  = false
tic
StripAnalysis(inputVideo, uint8(referenceFrame), parametersStructure);
toc

%%
% 'mex', 'fft', 'normxcorr'
% w/ and w/o GPU
%%method to use for cross-correlation.
%                                     you can choose from 'normxcorr' for
%                                     matlab's built-in normxcorr2, 'mex'
%                                     for opencv's correlation, 'fft'
%                                     for our custom-implemented fast
%                                     correlation method, or 'cuda' for our
%                                     custom-implemented cuda-based method.
%                                     (default 'mex')
