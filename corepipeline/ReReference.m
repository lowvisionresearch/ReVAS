function [eyePositionTraces_reRef, params, referenceFrame, timeArray, ...
    globalRef] = ReReference(globalRef, localRef, finalFilename, params)
% ReReference    	Update eyePositionTraces when given a global reference.
%   ReReference finds the position of a local reference frame relative to a
%   global reference frame. The row/column coordinates of the local 
%   reference frame are then added to the eyePositionTraces for the 
%   video that generated that localReferenceFrame.
%
%
%   globalRef is either the filepath to the globalRef or an array. localRef
%   is the reference frame resulting from calling FineRef. filename
%   is the output file from stripAnalysis. 

%% First initialize relevant variables

% If globalRef was passed in as a filepath, read the file.
if ischar(globalRef)
    % Leave this for now because the tiff file is not of the same type of
    % the localReferenceFrame 
    globalRef = imread(globalRef);
end

if ischar(localRef)
    localRef = imread(localRef);
end

% Load eyePositionTraces and timeArray from the output of stripAnalysis
load(finalFilename);

%% Check that inputs are valid
if size(globalRef, 1) < size(localRef, 1) || size(globalRef, 2)...
        < size(localRef, 2)
    
    yDifference = abs(size(globalRef, 1) - size(localRef, 1));
    xDifference = abs(size(globalRef, 2) - size(localRef, 2));
    
    globalRef = padArray(globalRef, [ceil(yDifference/2), ...
        ceil(xDifference/2)], 0, 'both');
end
%% Perform a cross-correlation on the global reference frame
% StripAnalysis only accepts 3D video input, so create a dummy frame and
% designate it as a bad frame.
dimensions = size(localRef);

threeDimensionalFrame = zeros(dimensions(1), dimensions(2), 2);
threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + localRef;
params.badFrames = 2;

% Check that strip height/width are set to the height of the full frame
params.stripHeight = dimensions(1);
params.stripWidth = dimensions(2);
[~, usefulEyePositionTraces, ~, ~] = StripAnalysis(threeDimensionalFrame, ...
    globalRef, params);

columnCoordinate = usefulEyePositionTraces(1, 1);
rowCoordinate = usefulEyePositionTraces(1, 2);
%% Add the local reference shift to eyePositionTraces 
eyePositionTraces(:, 1) = eyePositionTraces(:, 1) + columnCoordinate;
eyePositionTraces(:, 2) = eyePositionTraces(:, 2) + rowCoordinate;

%% Reassign variables and save.
eyePositionTraces_reRef = eyePositionTraces;
referenceFrame = localRef;
outputFileName = finalFilename;
outputFileName((end-3):end) = [];
outputFileName(end+1:end+6) = '_reref';
save(outputFileName, 'eyePositionTraces_reRef', 'timeArray', ...
        'params', 'referenceFramePath', 'globalRef');
%% Display the results
drawnow;  
hFig = figure;
hAx  = axes;
imshow(globalRef,'Parent', hAx);
imrect(hAx, [columnCoordinate, rowCoordinate, size(localRef,2), size(localRef,1)]);
figure('Name', 'ReferenceFrame')
imshow(localRef)


end