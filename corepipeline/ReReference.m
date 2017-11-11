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

%% Set parameters to defaults if not specified.

% If globalRef was passed in as a filepath, read the file.
if ischar(globalRef)
    % Leave this for now because the tif file is not of the same type of
    % the localReferenceFrame 
    globalRef = imread(globalRef);
end

if ischar(localRef)
    localRef = imread(localRef);
end

% Load eyePositionTraces and timeArray from the output of stripAnalysis
load(finalFilename);

if size(globalRef, 1) < size(localRef, 1) || size(globalRef, 2)...
        < size(localRef, 2)
    
    yDifference = abs(size(globalRef, 1) - size(localRef, 1));
    xDifference = abs(size(globalRef, 2) - size(localRef, 2));
    
    globalRef = padArray(globalRef, [ceil(yDifference/2), ...
        ceil(xDifference/2)], 0, 'both');
end

%% Perform a cross-correlation on the global reference frame

% This part is using the default method, which was found to be ineffective.
% c = normxcorr2(localRef,globalRef);
% [ypeak, xpeak] = find(c==max(c(:)));
% yoffSet = ypeak-size(localRef,1);
% xoffSet = xpeak-size(localRef,2);
% rowCoordinate = yoffSet;
% columnCoordinate = xoffSet;
% 
% % Check for tilt/distortion errors
% if yoffSet < 0 || xoffSet < 0 || xoffSet >= size(globalRef, 2) || ...
%         yoffSet >= size(globalRef, 1)
%     
%     if ~isfield(params, 'degreeRange')
%         params.degreeRange = -1:0.1:1;
%     end
%     disp('here')
%     for k = 1:max(size(params.degreeRange))
%         rotation = params.degreeRange(k);
%         tempLocalRef = imrotate(localRef, rotation);
%         
%         c = normxcorr2(tempLocalRef,globalRef);
%         [ypeak, xpeak] = find(c==max(c(:)));
%         yoffSet = ypeak-size(localRef,1);
%         xoffSet = xpeak-size(localRef,2);
%         
%         [surfX,surfY] = meshgrid(1:size(c,2), 1:size(c,1));
%         surf(surfX, surfY, c, 'linestyle', 'none');
%         xlim([1 size(c,2)]);
%         ylim([1 size(c,1)]);
%         zlim([-1 1]);
%         
%         % Mark the identified peak on the plot with an arrow.
%         text(xpeak, ypeak, max(c(:)), '\downarrow', 'Color', 'red', ...
%             'FontSize', 20, 'HorizontalAlignment', 'center', ...
%             'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
%         
%         drawnow;
%         
%         if yoffSet < 0 || xoffSet < 0 || xoffSet >= size(globalRef, 2) || ...
%         yoffSet >= size(globalRef, 1)
%             continue
%         else
%             rowCoordinate = yoffSet;
%             columnCoordinate = xoffSet;
%         end
%     end
% end

% StripAnalysis only accepts 3D video input, so create a dummy frame and
% designate it as a bad frame.
% PREVIOUS STUFF USING MATT"S FUNCTION GOES HERE dimensions = size(localRef);
dimensions = size(localRef);
threeDimensionalFrame = zeros(dimensions(1), dimensions(2), 2);
threeDimensionalFrame(:, :, 1) = threeDimensionalFrame(:, :, 1) + double(localRef);
params.badFrames = 2;

% Check that strip height/width are set to the height of the full frame
params.stripHeight = dimensions(1);
params.stripWidth = dimensions(2);

params.enableVerbosity = 1;
[~, usefulEyePositionTraces, ~, statisticsStructure] = ...
    StripAnalysis(threeDimensionalFrame, globalRef, params);

% Negate?
%usefulEyePositionTraces = -usefulEyePositionTraces;

columnCoordinate = usefulEyePositionTraces(1, 1);
rowCoordinate = usefulEyePositionTraces(1, 2);
peakRatio = statisticsStructure.peakRatios(1);

% Check for bad peaks
if peakRatio >= params.rotateMaximumPeakRatio
    params.reRef = 1;
    params.enableVerbosity = 1;
    [~, coordinatesAndDegrees] = RotateCorrect(threeDimensionalFrame, ...
        threeDimensionalFrame, globalRef, '.', params);
end

if exist('coordinatesAndDegrees', 'var') 
    if coordinatesAndDegrees(1, 1) > 0 && coordinatesAndDegrees(1, 2) > 0 &&...
            coordinatesAndDegrees(1, 1) < size(globalRef, 2) && ...
            coordinatesAndDegrees(1, 2) < size(globalRef, 1)
        columnCoordinate = coordinatesAndDegrees(1, 1);
        rowCoordinate = coordinatesAndDegrees(1, 2);
        disp(coordinatesAndDegrees(1, :))
    end
end

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