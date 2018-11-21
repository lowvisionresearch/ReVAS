function outputImage = MakeBigMontage(foldername, verbosity)
%MakeBigMontage
% Makes a global reference frame out of existing local reference frames,
% extracted from different retinal videos. The function takes a folder name
% as the input, searches for all files which has "coarseref" phrase in
% their name, reads the "referenceimage" array from each file and tiles
% them on a single global one.
%
% usage:
% outputImage = MakeBigMontage(foldername, verbosity)
% 
% foldername: name of the folder within which coarse reference frames
% reside.
% verbosity: set to 1 to see the output image, set to 2 to see the
% progress, otherwise set to 0.
% outputImage: a 2D uint8 array which represents the global reference frame
%
%
% Date          Author                 Change Log
% -------------------------------------------------
% 4/17/2017     mna                    Initial version
%
%
close all force;
clc;

if nargin<1
    foldername = uipickfiles;
    if ~iscell(foldername)
        if foldername == 0 
            fprintf('User cancelled folder selection. Silently exiting...\n');
            return;
        end
        foldername = {foldername};
    end
end

if nargin < 2
    verbosity = 2;
end

if verbosity
    fh = figure;
end

% keep a record of how many files contributed to the global reference frame
isDone = 0;
peakVal = [];
bestTilt = [];
peakValueThreshold = 0.35;
minPeakValue = 0.1;
repeatLimit = 5;
isSolveTilt = 1;
isManual = 0;

fileCounter = 0;
for folders=1:length(foldername)

    % if more than one folder is selected, get one folder name at a
    % time.
    currentFolder = foldername{folders};

    % list current folder's contents
    listing = dir(currentFolder);

    % get indices of all files (but not folders)
    indices = find(~[listing.isdir]); 

    % loop over all files in the selected folder
    for i=1:length(indices)

        % get next file name and number
        justname = listing(indices(i)).name;
        filename = lower([currentFolder filesep justname]);
        
        % Process only if it is a mat file that contains the keyphrase 'coarseref'
        if ~isempty(strfind(filename,'.mat')) && ...
                ~isempty(strfind(filename,'coarseref'))
            
            ind = strfind(justname,'_');
            if isempty(ind)
                ind = strfind(justname,' ');
            end
            if isempty(ind)
                ind = strfind(justname,'coarseref');
            end
            
            fileNo(i) = str2double(justname(1:ind(1)-1));
            fileCounter = fileCounter + 1;
            filelist{fileCounter} = filename;  
            
        end
    end

end

[~,sortOrder] = sort(fileNo);
filelist = filelist(sortOrder);

for i=1:length(filelist)
    fprintf('%s is added to the list.\n',filelist{i});
end
fprintf('A total of %d file will be used.\n',fileCounter);


% % find pairwise correlations
% resizeFactor = .25;
% correlationMatrix = zeros(length(filelist));
% for i=1:length(filelist)
%     img1 = load(filelist{i},'referenceimage');
%     parfor j=i+1:length(filelist)
%         img2 = load(filelist{j},'referenceimage');
%         [~,correlationMatrix(i,j)] =  LocalizeRef(...
%             imresize(img1.referenceimage,resizeFactor),...
%             imresize(img2.referenceimage,resizeFactor));
%     end
% end



% repeat going over each file because, a badly correlated image, might
% become nicely correlated after constructing the global reference a
% little further.. espeically considering that this is a big montage
% function, this is necessary.
repeat = 0;
contributingFiles = {};

while sum(isDone == 0)>0 && (repeat < repeatLimit)

    peakValueThreshold = peakValueThreshold / (repeat*0.1+1);
    if peakValueThreshold < minPeakValue
        peakValueThreshold = minPeakValue;
    end

    fileCounter = 0;
    % loop over all files 
    for j=1:1:length(filelist)

        try
            % get next file name
            filename = filelist{j};

            % load the local reference frame
            load(filename,'referenceimage');
            referenceimage = double(referenceimage);
%             referencematrix = double(referencematrix);

            % replace nans with zeros
            referenceimage(isnan(referenceimage)) = 0;

            % increment the counter
            fileCounter = fileCounter + 1;
            isDone(j) = 0;

            % for the first contributing file, use its reference frame as
            % the initial seed for the global one. Otherwise, do the
            % montage.
            if fileCounter == 1 && repeat==0
                globalRef = referenceimage;%squeeze(referencematrix); %#ok<NODEF>
                paddedIndices = globalRef < 1;
                counterRef = ones(size(globalRef))-paddedIndices;
                peakVal(1) = 0.4;
                isDone(j) = 1;
            else

                if ~isDone(j)
                    % estimate the offsets to localize the local ref on global
                    % one.
                    counterRef(counterRef == 0) = 1;
                    [offsets, peakVal(j), bestTilt(j)] = ...
                        LocalizeRef( (referenceimage), ...
                         (globalRef./counterRef),isSolveTilt); %#ok<*ASGLU>

                    % use this local frame only if we can localize it on the
                    % global ref. without any problem.
                    if sum(isnan(offsets)) == 0 && peakVal(j)> peakValueThreshold

%                         referenceimage = squeeze(referencematrix(:,:,3)); %#ok<NODEF>
                       
                        if isSolveTilt
                            referenceimage = imrotate(uint8(referenceimage),...
                            bestTilt(j),'bilinear','crop'); %#ok<*UNRCH>
                        end
                        
                        paddedIndices = referenceimage < 1;

                        % put the local ref on global ref and increment the counter
                        [globalRef, counterRef] = ...
                            UpdateRef(referenceimage,globalRef,...
                            counterRef,offsets,paddedIndices);

                        % mark this file as done.
                        isDone(j) = 1;
                    else
                        
                        if isManual
                            inp = uint8((referenceimage));
                            ref = uint8((globalRef./counterRef));
                            [movingPoints, fixedPoints] = cpselect(inp,ref,'Wait',true);
                            movingPoints = cpcorr(movingPoints,fixedPoints,inp,ref);
                            tform = fitgeotrans(movingPoints,fixedPoints,'Similarity');
                            rref = imref2d(size(ref));
                            regs = imwarp(inp,tform,'OutputView',rref);
%                             figure(888); cla; imshowpair(regs,ref,'blend');
%                             offsets = round([tform.T(3,2) tform.T(3,1)]);
%                             bestTilt = acosd(tform.T(1,1)/sqrt(sum(tform.T(1:2,1).^2)));
%                             peakVal = 0.4;
                            paddedIndices = regs < 1;
                            counterRef(~paddedIndices) = counterRef(~paddedIndices) + 1;
                            globalRef = globalRef + double(regs);
                            isDone(j) = 1;
                            
                        else
                            % since we did not use the current file
                            fileCounter = fileCounter - 1;
                        end
                    end
                end
            end

            % show progress 
            if verbosity == 2
                figure(fh);
                imshow(uint8(globalRef./counterRef));
                drawnow;
                figure(1234);
                subplot(2,1,1)
                cla; plot(peakVal,'-r.');hold on;
                plot(ones(1,length(filelist))*peakValueThreshold,'-k');
                subplot(2,1,2)
                cla; plot(bestTilt,'-b.');
            end

        catch err 
            err.message
            err.stack.line
            err.stack.name
            fprintf('Local reference frame does not exist, skipping the file\n');
            continue;
        end

    end

    repeat = repeat + 1;
    fprintf('Repeat number: %d\n',repeat);
end

fprintf('%d files contributed to the final reference frame from\nthe folder %s\n',...
    fileCounter,currentFolder);


% compute the output image
outputImage = uint8(globalRef ./ counterRef);

% show it to the user if verbosity set to a nonzero value
if verbosity
    figure(fh);
    imshow(outputImage);
end






function [offsets, peakVal, bestTilt]...
    = LocalizeRef(referenceimage, globalRef,isSolveTilt)

try
    if isSolveTilt
        [newLocalRef, bestTilt] = ...
            SolveTiltIssue(referenceimage,globalRef,0.5);
    else
        bestTilt = 0;
        newLocalRef = referenceimage;
    end

    N = [0 0];
    c = normxcorr2(newLocalRef,globalRef);
catch err1 %#ok<*NASGU>
    % probably complaining about size, so zeropad the ref and see how it
    % goes. don't forget to control for the additional offsets this causes.
    N = size(referenceimage)-size(globalRef);
    N(N<0) = 0;
    
    tempRef = padarray(globalRef,N);
    try
        if isSolveTilt
            [newLocalRef, bestTilt] = ...
                SolveTiltIssue(referenceimage,tempRef,1.0);
        else
            bestTilt = 0;
            newLocalRef = referenceimage;
        end
        
        c = normxcorr2(newLocalRef,tempRef);
    catch err2
        % okay, either something is really wrong or the size mismatch is
        % really huge. whatever the cause, if the program comes here,
        % return with NaNs and let the user know.
        fprintf('Problem with normxcorr2, probably huge mismatch between ref. sizes.\n');
        offsets = [NaN NaN];
        return;
    end
end

% Find the peak by the smoothing method.
[ypeak, xpeak, peakVal] = FindPeak(c,N);

yoffset = ypeak-size(referenceimage,1)-N(1);
xoffset = xpeak-size(referenceimage,2)-N(2);
offsets = gather([yoffset, xoffset]);
peakVal = gather(peakVal);


    

function [R,T] = rot3dfit(X,Y)
%ROT3DFIT Determine least-square rigid rotation and translation.
% [R,T,Yf] = ROT3DFIT(X,Y) permforms a least-square fit for the
% linear form
%
% Y = X*R + T
%
% where R is a 3 x 3 orthogonal rotation matrix, T is a 1 x 3
% translation vector, and X and Y are 3D points sets defined as
% N x 3 matrices. Yf is the best-fit matrix.
%
% See also SVD, NORM.
%
% rot3dfit: Frank Evans, NHLBI/NIH, 30 November 2001
%
% ROT3DFIT uses the method described by K. S. Arun, T. S. Huang,and
% S. D. Blostein, "Least-Squares Fitting of Two 3-D Point Sets",
% IEEE Transactions on Pattern Analysis and Machine Intelligence,
% PAMI-9(5): 698 - 700, 1987.
%
% A better theoretical development is found in B. K. P. Horn,
% H. M. Hilden, and S. Negahdaripour, "Closed-form solution of
% absolute orientation using orthonormal matrices", Journal of the
% Optical Society of America A, 5(7): 1127 - 1135, 1988.
%
% Special cases, e.g. colinear and coplanar points, are not
% implemented.

narginchk(2,2);
if size(X,2) ~= 3, error('X must be N x 3'); end;
if size(Y,2) ~= 3, error('Y must be N x 3'); end;
if size(X,1) ~= size(Y,1), error('X and Y must be the same size'); end;

% mean correct
Xm = mean(X,1); X1 = X - ones(size(X,1),1)*Xm;
Ym = mean(Y,1); Y1 = Y - ones(size(Y,1),1)*Ym;

% calculate best rotation using algorithm 12.4.1 from
% G. H. Golub and C. F. van Loan, "Matrix Computations"
% 2nd Edition, Baltimore: Johns Hopkins, 1989, p. 582.

XtY = (X1')*Y1;
[U,S,V] = svd(XtY);
R = U*(V');

% solve for the translation vector
T = Ym - Xm*R;

% % calculate fit points
% Yf = X*R + ones(size(X,1),1)*T;
% 
% % calculate the error
% dY = Y - Yf;
% Err = norm(dY,'fro'); % must use Frobenius norm



function [ypeak, xpeak, maxVal] = FindPeak(c,N)

% for robust and fast peak finding, we smooth the crosscorrelation surface
% and subtract from the original one, and find the max point on the
% resultant map.
% Initially, I used a median filter based approach but it is very labor
% intensive. Gauss filtering works very fast.
temp = c - imgaussfilt(c,21);

if nargin<2
    maxVal = max(temp(:));
    [ypeak, xpeak] = find(temp == maxVal);
else
    % if the second argument is provided, limit the peak finding to a
    % small window defined by the central portion of the correlation map.
    st = N(1)*1+100;
    en = N(2)*1+100;
    tempC = temp(st:end-en,...
              st:end-en);
    maxVal = max(tempC(:));
    [ypeak, xpeak] = find(tempC == maxVal);
    ypeak = ypeak + st - 1;
    xpeak = xpeak + st - 1;
    
%     maxVal = c(ypeak,xpeak);
end


% tilt issue solver
function [inp, bestTilt] = SolveTiltIssue(inp,ref,inc)

% rotate the input image, crosscorrelate with ref, and find the peak.
% Across all tilts, find the best one which results in the max peak.
% Also rotate the input image based on the bestTilt.

tilt = -5:inc:5; % in degrees
for j=1:length(tilt)
    rotated = imrotate(uint8(inp),tilt(j),'bilinear','crop');
%     rotated = PadNoise(rotated);
    c = normxcorr2(double(rotated),double(ref));
    [~, ~, maxVal(j)] = FindPeak(c);
end

interpolatedVal = interp1(tilt,maxVal,-10:.1:10);
bestTilt = max(interpolatedVal);
% [~,ind] = max(maxVal);
% bestTilt = tilt(ind);
inp = imrotate(uint8(inp),bestTilt,'bilinear','crop');
% inp = uint8(PadNoise(inp));


function output = PadNoise(input)

% add noise to zero-padded regions
input = double(input);
padIndices = input < 1;
noiseFrame = padIndices.*(rand(size(padIndices),' ')*100+mean(input(:)));
output = double(input) + noiseFrame;



function [globalRef, counterRef] = ...
    UpdateRef(referenceimage,globalRef,counterRef,offsets,paddedIndices)

% if offsets are negative or beyond the size of the globalRef, zeropad it.
for i=1:2
    if offsets(i)<0
        N(i) = abs(offsets(i)); %#ok<*AGROW>
        postPad(i) = 0;
    else
        N(i) = 0;
        postPad(i) = offsets(i);
    end
end

% do the zero padding
globalRef = padarray(globalRef,N,'pre');
counterRef = padarray(counterRef,N,'pre');

% get the size of the local ref.
[locM, locN] = size(referenceimage);

% compute how large the local ref is 
sizeDifference = [locM locN] + postPad - size(globalRef);

% if bigger than the global ref, make global ref bigger. Note that we don't
% need to compensate for this because the padding will be done to the end
% of the reference array.
if sum(sizeDifference>0)>0
    sizeDifference(sizeDifference<0) = 0;
    globalRef = padarray(globalRef,sizeDifference,'post');
    counterRef = padarray(counterRef,sizeDifference,'post');
end

% coordinates of the top-left corner of the local ref. on the global ref.
coords = offsets + N + 1;

try
    % now update the global ref. and the counter
    globalRef(coords(1):coords(1)+locM-1, coords(2):coords(2)+locN-1) = ...
        globalRef(coords(1):coords(1)+locM-1, coords(2):coords(2)+locN-1) + ...
        double(referenceimage);
    counterRef(coords(1):coords(1)+locM-1, coords(2):coords(2)+locN-1) = ...
        counterRef(coords(1):coords(1)+locM-1, coords(2):coords(2)+locN-1) + ...
        ones(locM,locN) - paddedIndices;
catch err
    err;
end

% replace NaNs with zeros.
globalRef(isnan(globalRef)) = 0;
counterRef(isnan(counterRef)) = 0;



    
