function stats = Timer_StripAnalysis


%% read in sample video

% for tester, we're using an aoslo video, because tslo or larger fov
% videos usually need to be preprocessed to remove within-frame
% nonhomogeneities so that cross-correlation will be robust. 

% the video resides under /testing folder.
inputVideo = FindFile('aoslo.avi');

% test with a video array
videoArray = ReadVideoToArray(inputVideo);

n = gpuDeviceCount;
t.time = [];
t.cond = [];
t = repmat(t,11,1);
figure('name','Timer_StripAnalysis','units','normalized','outerposition',...
    [.1 .1 .8 .8]);
i = 0;

%% normxcorr
tic;
i = i + 1;
p = struct; 
p.overwrite = true;
p.enableVerbosity = 0;
p.minPeakThreshold = 0.4;
p.maxMotionThreshold = 0.3;
p.adaptiveSearch = false;
p.dynamicReference = false;
p.corrMethod = 'normxcorr';
[pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
t(i).time = toc;
t(i).cond = p.corrMethod;
ShowResults(t,i,timeSec,pos)

%% normxcorr-adaptive
tic;
i = i + 1;
p.enableGPU = false;
p.adaptiveSearch = true;
[pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
t(i).time = toc;
t(i).cond = [p.corrMethod '-adaptive'];
ShowResults(t,i,timeSec,pos)

%% normxcorr-gpu
tic;
i = i + 1;
p.enableGPU = true;
p.adaptiveSearch = false;
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = [p.corrMethod '-gpu'];
ShowResults(t,i,timeSec,pos)

%% normxcorr-gpu-adaptive
tic;
i = i + 1;
p.enableGPU = true;
p.adaptiveSearch = true;
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = [p.corrMethod '-gpu-adaptive'];
ShowResults(t,i,timeSec,pos)

%% fft
tic;
i = i + 1;
p.adaptiveSearch = false;
p.enableGPU = false;
p.corrMethod = 'fft';
[pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
t(i).time = toc;
t(i).cond = p.corrMethod;
ShowResults(t,i,timeSec,pos)

%% fft-gpu
tic;
i = i + 1;
p.enableGPU = false;
p.corrMethod = 'fft';
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = [p.corrMethod '-gpu'];
ShowResults(t,i,timeSec,pos)

%% mex
tic;
i = i + 1;
p.corrMethod = 'mex';
[pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
t(i).time = toc;
t(i).cond = p.corrMethod;
ShowResults(t,i,timeSec,pos)

%% mex-gpu
tic;
i = i + 1;
p.corrMethod = 'mex';
p.enableGPU = true;
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = [p.corrMethod '-gpu'];
ShowResults(t,i,timeSec,pos)

%% mex-adaptive
tic;
i = i + 1;
p.corrMethod = 'mex';
p.enableGPU = false;
p.adaptiveSearch = true;
[pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
t(i).time = toc;
t(i).cond = [p.corrMethod '-adaptive'];
ShowResults(t,i,timeSec,pos)

%% mex-gpu-adaptive
tic;
i = i + 1;
p.corrMethod = 'mex';
p.enableGPU = true;
p.adaptiveSearch = true;
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = [p.corrMethod '-gpu-adaptive'];
ShowResults(t,i,timeSec,pos)

%% cuda
tic;
i = i + 1;
p.adaptiveSearch = false;
p.corrMethod = 'cuda';
if n > 0
    [pos, timeSec, rawPos, peakVals, q] = StripAnalysis(videoArray, p); %#ok<*ASGLU>
    t(i).time = toc;
else
    pos = nan;
    timeSec = nan;
    t(i).time = nan;
end
t(i).cond = p.corrMethod;
ShowResults(t,i,timeSec,pos)


%% show results
[~,ix] = sort([t.time]);
stats = struct2table(t(ix));


function ShowResults(t,num,timeSec,pos)

disp(t(num))
subplot(3,4,num)
plot(timeSec,pos,'.','linewidth',1.5);
xlabel('time (sec)')
ylabel('position (px)')
title(t(num).cond)
drawnow;
