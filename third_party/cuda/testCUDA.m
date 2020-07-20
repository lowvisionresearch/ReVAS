clearvars
close all
clc

% initialize gpu
g = gpuDevice(1);
reset(g);

% find a demo video
vid = VideoReader(FindFile('aoslo.avi'));

% get first frame
referenceFrame = single(readFrame(vid));

% set up strip params
stripHeight = 11;
stripWidth = 512;

% get a new frame (a little away from current frame)
newFrame = readFrame(vid);

% cuda prep
cuda_prep(referenceFrame,stripHeight,stripWidth,true)

% to save difference stats between Matlab's normxcorr2 based method and
% cuda method.
difference = nan*ones(16,3);

% correlation map size
outsize = size(referenceFrame) + [stripHeight-1 stripWidth-1];

% Go over strips and localize strips on reference frame
i = 1;
for startx = 1
    for starty = 1:10:51
        strip = single(newFrame(starty:starty+stripHeight-1,startx:startx+stripWidth-1));
        [corrmap0,xloc0,yloc0,peak00,peak01] = cuda_match(strip,false); % faster
        [corrmap,xloc,yloc,peak0,peak1] = cuda_match(strip,true); % slower
        
        correlationMap = fftshift(corrmap);
        correlationMap = correlationMap(1:outsize(1),1:outsize(2));
        [val, ind] = max(correlationMap(:));
        [a0,b0] = ind2sub(size(correlationMap),ind);
        fprintf('*%d, %d: %f\n',b0,a0,val);
        
        gold = normxcorr2(strip,referenceFrame);
        [val, ind] = max(gold(:));
        [a,b] = ind2sub(size(gold),ind);
        goldmax = sort(gold(:),'descend');
        fprintf('**%d, %d: %f\n',b,a,goldmax(1));
        
        difference(i,1) = a0/a;
        difference(i,2) = b0/b;   
        difference(i,3) = mean(abs(correlationMap(:)-gold(:)));
        difference(i,4) = max(abs(correlationMap(:)-gold(:)));
        i = i+1;
        
        figure(1);
        cla;
        mesh([correlationMap gold]);
        caxis([-1 1])
    end
end


figure; plot(difference,'.');
nanmean(difference)


