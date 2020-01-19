clear all
close all
clc

%%
g = gpuDevice(1);
%reset(g);
% diary('log');
vid = VideoReader('cuda/sample.avi');
% vid = VideoReader('ExampleIVideosForTesting/Wang_PRLVideo_20158R_V012.avi');

referenceFrame = 125*single(ones(1024,1024));
referenceFrame(1:512,1:512) = single(readFrame(vid));
%referenceFrame = referenceFrame(1:70,1:50);

stripHeight = 11;
stripWidth = 512;

startx = 10;
starty = 30;
strip1 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);

startx = 1;
starty = 20;
strip2 = referenceFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1);

figure(); imshow(strip2,[]);
%%
mexcuda -lcufft cuda_match.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu
%%
cuda_prep(referenceFrame,stripHeight,stripWidth,true)
%%
cuda_match(strip2,true);
%%
newFrame = readFrame(vid);
%%
cuda_prep(referenceFrame,stripHeight,stripWidth,true)
i = 1;
difference = nan*ones(16,3);
outsize = size(normxcorr2(strip1,referenceFrame));
for startx = 1
    for starty = 1:10:51
        strip0 = single(newFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1));
        strip1 = 255*imnoise(strip0/255,'gaussian');
        [corrmap0,xloc0,yloc0,peak00,peak01] = cuda_match(strip1,false);
        [corrmap,xloc,yloc,peak0,peak1] = cuda_match(strip1,true);
        
        correlationMap = fftshift(corrmap);
        correlationMap = correlationMap(1:outsize(1),1:outsize(2));
        %figure; imshow(correlationMap,[]);
        [val, ind] = max(correlationMap(:));
        [a0,b0] = ind2sub(size(correlationMap),ind);
        %corrmax = sort(correlationMap(:),'descend');
        fprintf('*%d, %d: %f\n',b0,a0,val);
        
        gold = normxcorr2(strip1,referenceFrame);
        %figure; imshow(gold,[]);
        [val, ind] = max(gold(:));
        [a,b] = ind2sub(size(gold),ind);
        goldmax = sort(gold(:),'descend');
        %a = goldmax(1);
        %b = goldmax(2);
        fprintf('**%d, %d: %f\n',b,a,goldmax(1));
        
        %figure(); imshow(correlationMap-gold);
        
        difference(i,1) = a0/a;
        difference(i,2) = b0/b;   
        difference(i,3) = mean(abs(correlationMap(:)-gold(:)));
        difference(i,4) = max(abs(correlationMap(:)-gold(:)));
        i = i+1;
    end
end
figure; plot(difference,'.');
nanmean(difference)
%%
vid = VideoReader('sample.avi');
referenceFrame = single(readFrame(vid));
p.stripHeight = 11;
p.stripWidth = 512;

starty = 30;
strip1 = referenceFrame(starty:starty+p.stripHeight-1,:);

p.adaptiveSearch = false;
p.badFrames = false;
p.referenceFrame = referenceFrame;
p.stripHeight = 11;
refsize = size(referenceFrame);
p.stripWidth = refsize(1);
p.overwrite = true; 
p.rowStart = 1;
p.rowEnd = size(referenceFrame,1);

p.enableGPU = true;
p.copyMap = true;
p.corrMethod  = 'cuda';
cuda_prep(p.referenceFrame,p.stripHeight,p.stripWidth,true);
p.outsize =  size(p.referenceFrame)+[p.stripHeight-1,p.stripWidth-1];
p.copyMap = true; % note: this slows things down considerably, much faster to just use peaks

[correlationMap, xPeak, yPeak, peakValue, ~] = ...
          LocateStrip(strip1,p,struct);
%%
params2 = p;
params2.corrMethod = 'normxcorr';
 [correlationMap2, xPeak2, yPeak2, peakValue2, ~] = ...
                    LocateStrip(strip1,params2,struct);
%%
figure(); plot(peakPosition,'.'); figure(); plot(peakValueArray,'.'); figure(); plot(position,'.'); figure(); plot(rawPosition,'.');
%%
% timing other methods:
%$
g = gpuDevice(1);
vid = VideoReader('cuda/sample.avi');
referenceFrame = readFrame(vid);
inputVideo = 'cuda/sample.avi';
p.scalingFactor = 1;
p.adaptiveSearch = false;
p.downSampleFactor = 1;
p.badFrames = logical([]);
p.referenceFrame = referenceFrame;
p.stripHeight = 11;
refsize = size(referenceFrame);
p.stripWidth = refsize(1);
p.overwrite = true; 
%%
reset(g);
p.enableGPU = true;
p.copyMap = false;
p.corrMethod  = 'cuda'
p.enableVerbosity = 0;
tic
StripAnalysis(inputVideo, p);
toc
%%
reset(g);
p.enableGPU = true;
p.copyMap = true;
p.dynamicReference = true;
p.enableVerbosity = 4;
p.corrMethod  = 'cuda'
tic
StripAnalysis(inputVideo, p);
toc
%%
p.enableGPU = false;
p.dynamicReference = true;
p.enableVerbosity = 4;
p.corrMethod  = 'fft'
tic
StripAnalysis(inputVideo, p);
toc
%%
reset(g);
p.corrMethod  = 'mex';
p.enableGPU  = true
tic
StripAnalysis(inputVideo, p);
toc
%%
reset(g);
p.corrMethod  = 'mex';
p.enableGPU  = false
tic
StripAnalysis(inputVideo, p);
toc
%%
reset(g);
p.corrMethod  = 'fft';
p.dynamicReference =true;
p.enableVerbosity = 4;
p.enableGPU  = false
tic
StripAnalysis(inputVideo, p);
toc
%%
vids = dir('ExampleIVideosForTesting/*.avi')

p.corrMethod  = 'normxcorr';
p.dynamicReference = true;
p.enableVerbosity = 0;
p.enableGPU  = false;
for v = 1:length(vids)
    tic
    StripAnalysis(vids(v).name, p);
    toc
end