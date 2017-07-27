function [eyePositionTraces_reRef, params, referenceFrame, timeArray, ...
    globalRef] = ReReference(globalRef, filename, params)
% ReReference    	Update eyePositionTraces when given a global reference.
%   ReReference finds the position of a local reference frame relative to a
%   global reference frame. The row/column coordinates of the local 
%   reference frame are then added to the eyePositionTraces for the 
%   video that generated that localReferenceFrame.
%
%
%   globalRef is either the filepath to the globalRef or an array. filename
%   is the output file from stripAnalysis. 

%% First initialize relevant variables

% If globalRef was passed in as a filepath, read the file.
if ischar(globalRef)
    % Leave this for now because the tiff file is not of the same type of
    % the localReferenceFrame ONLY DO THIS IF THE FILE IS NOT THE TYPE OF
    % INTEGER WE WANT
    globalRef = double(imread(globalRef))/255;
end

% Load eyePositionTraces and timeArray from the output of stripAnalysis
load(filename)
%localReferenceFrame = imread(referenceFrame);
localReferenceFrame = params.referenceFrame;

%% Check that inputs are valid
if size(globalRef, 1) < size(localReferenceFrame, 1) || size(globalRef, 2)...
        < size(localReferenceFrame, 2)
    error('Global reference frame is smaller than local reference frame');
end
    
%% Trim the local reference frame if there is padding in the corners

% First check for stray columns/rows that are entirely 0
k = 1;
while k<=size(localReferenceFrame, 1)
    if localReferenceFrame(k, :) == 0
        localReferenceFrame(k, :) = [];
        continue
    end
    k = k + 1;
end

k = 1;
while k<=size(localReferenceFrame, 2)
    if localReferenceFrame(:, k) == 0
        localReferenceFrame(:, k) = [];
        continue
    end
    k = k + 1;
end

% Now cut off the margin columns
numPixelsInColumn = size(localReferenceFrame, 1);
zeroIndices = find(localReferenceFrame==0);

% Doing this next line will get the column numbers of the zeros, since
% zeroIndices is a column vector and its indices are not in terms of
% row-column
zeroIndices = ceil(zeroIndices/numPixelsInColumn);

for k = 1:max(size(zeroIndices))
    if max(size(zeroIndices)) == 1
        startColumn = 1;
        endColumn = size(localReferenceFrame, 2);
        break
    elseif k + 1 > max(size(zeroIndices))
        if ~exist('startColumn', 'var')
            startColumn = 1;
        end
        endColumn = size(localReferenceFrame, 2);
        break
    elseif zeroIndices(k+1) == zeroIndices(k) || zeroIndices(k+1) == ...
            zeroIndices(k) + 1
        continue
    else
        startColumn = zeroIndices(k) + 1;
        endColumn = zeroIndices(k+1) - 1;
        break
    end
end

localReferenceFrame = localReferenceFrame(:, startColumn:endColumn);

%% Perform a cross-correlation on the global reference frame
% REMEMBER TO COME BACK AND ADD OPTIONS FOR TILT/STRETCH/DISTORTION
c = normxcorr2(localReferenceFrame, globalRef);
params.enableGaussianFiltering = 0;
params.stripHeight = size(localReferenceFrame,1);
[xPeak, yPeak, peakValue, ~] = FindPeak(c, params);
yoffSet = yPeak-size(localReferenceFrame,1);
xoffSet = xPeak-size(localReferenceFrame,2);
rowCoordinate = yoffSet + 1;
columnCoordinate = xoffSet + 1;

%% Add the local reference shift to eyePositionTraces 
eyePositionTraces(:, 1) = eyePositionTraces(:, 1) + columnCoordinate;
eyePositionTraces(:, 2) = eyePositionTraces(:, 2) + rowCoordinate;

%% Reassign variables and save.
eyePositionTraces_reRef = eyePositionTraces;
referenceFrame = localReferenceFrame;
outputFileName = filename;
outputFileName((end-3):end) = [];
outputFileName(end+1:end+6) = '_reref';
save(outputFileName, 'eyePositionTraces_reRef', 'timeArray', ...
        'params', 'referenceFramePath', 'globalRef');
%% Display the results
[surfX,surfY] = meshgrid(1:size(c,2), 1:size(c,1));
surf(surfX, surfY, c,'linestyle','none');
xlim([1 size(c,2)]);
ylim([1 size(c,1)]);
zlim([-1 1]);

% Mark the identified peak on the plot with an arrow.
text(xPeak, yPeak, peakValue, '\downarrow', 'Color', 'red', ...
    'FontSize', 20, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

drawnow;  
hFig = figure;
hAx  = axes;
imshow(globalRef,'Parent', hAx);
imrect(hAx, [columnCoordinate, rowCoordinate, size(localReferenceFrame,2), size(localReferenceFrame,1)]);
figure('Name', 'ReferenceFrame')
imshow(localReferenceFrame)


end