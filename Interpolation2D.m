function [interpolatedPixelCoordinates, errorStructure]...
    = interpolation2D(correlationMap2D, peakCoordinates, parametersStructure)
%2D INTERPOLATION Completes 2D Interpolation on a correlation map.
%   Completes 2D Interpolation on a correlation map and returns the new
%   interpolated peak coordinates. Uses the |spline| option in |interp2|.

%% Input Validation

if ~ismatrix(correlationMap2D)
    error('Invalid Input for interpolation2D (correlationMap2D is not 2 dimensional)');
elseif true(size(peakCoordinates) ~= [1 2])
    error('Invalid Input for interpolation2D (peakCoordinates is not 1x2 coordinate pair)');
elseif ~isfield(parametersStructure, 'neighborhoodSize')
    error('Invalid Input for interpolation2D (neighborhoodSize is not a field of parametersStructure)');
elseif ~isfield(parametersStructure, 'subpixelDepth')
    error('Invalid Input for interpolation2D (subpixelDepth is not a field of parametersStructure)');
elseif ~isscalar(parametersStructure.neighborhoodSize) || ...
        mod(parametersStructure.neighborhoodSize, 2) ~= 1
    error('Invalid Input for interpolation2D (neighborhoodSize is not an odd scalar)');
elseif ~isscalar(parametersStructure.subpixelDepth)
    error('Invalid Input for interpolation2D (subpixelDepth is not a scalar)');
end

%% Apply |interp2|
halfNeighborhoodSize = (parametersStructure.neighborhoodSize - 1) / 2;
[meshgridX, meshgridY] = meshgrid(-halfNeighborhoodSize:halfNeighborhoodSize);
meshgridX = meshgridX + peakCoordinates(1);
meshgridY = meshgridY + peakCoordinates(2);

% TODO Unsure if subpixelDepth is being interpretted correctly
% Trimming to neighborhoodSize
gridSpacing = parametersStructure.subpixelDepth / 100;
[finerMeshgridX, finerMeshgridY] = ...
    meshgrid(-halfNeighborhoodSize:gridSpacing:halfNeighborhoodSize);
finerMeshgridX = finerMeshgridX + peakCoordinates(1);
finerMeshgridY = finerMeshgridY + peakCoordinates(2);
correlationMap2D = correlationMap2D(...
    peakCoordinates(2)-halfNeighborhoodSize:peakCoordinates(2)+halfNeighborhoodSize,...
    peakCoordinates(1)-halfNeighborhoodSize:peakCoordinates(1)+halfNeighborhoodSize);

finerCorrelationMap2D = interp2(meshgridX, meshgridY, correlationMap2D,...
    finerMeshgridX, finerMeshgridY, 'spline');

%% Find new peak of new
% correlation map and calculate pixel coordinate.
[ypeak, xpeak] = find(finerCorrelationMap2D==max(finerCorrelationMap2D(:)));

% Scale back down to pixel units
interpolatedPixelCoordinates = [xpeak ypeak];
interpolatedPixelCoordinates = ...
    floor((interpolatedPixelCoordinates - 1) / gridSpacing^-1) + 1 + ...
    (mod((interpolatedPixelCoordinates - 1), gridSpacing^-1) * gridSpacing);

% Center back around original peak
interpolatedPixelCoordinates = interpolatedPixelCoordinates ...
    - 1 - halfNeighborhoodSize + peakCoordinates;

%% TODO Not implemented yet
errorStructure = struct();
end


