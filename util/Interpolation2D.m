function [interpolatedPixelCoordinates, errorStructure]...
    = Interpolation2D(correlationMap2D, peakCoordinates, parametersStructure)
%2D INTERPOLATION Completes 2D Interpolation on a correlation map.
%   Completes 2D Interpolation on a correlation map and returns the new
%   interpolated peak coordinates. Uses the |spline| option in |interp2|.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   neighborhoodSize                : the length of one of the sides of the
%                                     neighborhood area over which
%                                     interpolation is to be performed over
%                                     in pixels. (default 7)
%   subpixelDepth                   : the scaling of the desired level of
%                                     subpixel depth. (default 2)

%% Input Validation

try
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
    
    errorStructure = struct();
catch MException
    errorStructure = struct(MException);
end

if size(correlationMap2D, 1) == 1 || size(correlationMap2D, 2) == 1
    RevasWarning(['Interpolation not applied this iteration since correlationMap2D dimensions were ' int2str(size(correlationMap2D, 1)) ' x ' int2str(size(correlationMap2D, 2))]);
    interpolatedPixelCoordinates = peakCoordinates;
    return;
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'neighborhoodSize')
   parametersStructure.neighborhoodSize = 7;
   RevasMessage('using default parameter for neighborhoodSize');
else
   parametersStructure.neighborhoodSize = parametersStructure.neighborhoodSize;
   if ~IsNaturalNumber(parametersStructure.neighborhoodSize)
       error('neighborhoodSize must be a natural number');
   end
end
if ~isfield(parametersStructure, 'subpixelDepth')
   parametersStructure.subpixelDepth = 2;
   RevasMessage('using default parameter for subpixelDepth');
else
   parametersStructure.subpixelDepth = parametersStructure.subpixelDepth;
   if ~IsPositiveRealNumber(parametersStructure.subpixelDepth)
       error('subpixelDepth must be a positive, real number');
   end
end

%% Apply |interp2|
halfNeighborhoodSize = (parametersStructure.neighborhoodSize - 1) / 2;
[meshgridX, meshgridY] = meshgrid(-halfNeighborhoodSize:halfNeighborhoodSize);
meshgridX = meshgridX + peakCoordinates(1);
meshgridY = meshgridY + peakCoordinates(2);

% Trimming to neighborhoodSize
gridSpacing = 2^(-parametersStructure.subpixelDepth);
[finerMeshgridX, finerMeshgridY] = ...
    meshgrid(-halfNeighborhoodSize:gridSpacing:halfNeighborhoodSize);
finerMeshgridX = finerMeshgridX + peakCoordinates(1);
finerMeshgridY = finerMeshgridY + peakCoordinates(2);

% Dealing with if the window defined by neighborhood size is too large
% because we are at the border of the correlation map.
trimYLow = (peakCoordinates(1)-1-halfNeighborhoodSize) * -1;
trimXLow = (peakCoordinates(2)-1-halfNeighborhoodSize) * -1;
trimYHigh = peakCoordinates(1)+halfNeighborhoodSize - size(correlationMap2D, 1);
trimXHigh = peakCoordinates(2)+halfNeighborhoodSize - size(correlationMap2D, 2);

% Clear trim variables if no trimming necessary.
if trimYLow < 1
    trimYLow = 0;
end
if trimXLow < 1
    trimXLow = 0;
end
if trimYHigh < 0
    trimYHigh = 0;
end
if trimXHigh < 0
    trimXHigh = 0;
end

% Apply trimming to meshgrids as necessary.
meshgridX = meshgridX(trimYLow+1:end-trimYHigh, trimXLow+1:end-trimXHigh);
meshgridY = meshgridY(trimYLow+1:end-trimYHigh, trimXLow+1:end-trimXHigh);
finerMeshgridX = finerMeshgridX((trimYLow+1-1)*gridSpacing^-1 + 1:end-((trimYHigh+1-1)*gridSpacing^-1),...
    (trimXLow+1-1)*gridSpacing^-1 + 1:end-((trimXHigh+1-1)*gridSpacing^-1));
finerMeshgridY = finerMeshgridY((trimYLow+1-1)*gridSpacing^-1 + 1:end-((trimYHigh+1-1)*gridSpacing^-1),...
    (trimXLow+1-1)*gridSpacing^-1 + 1:end-((trimXHigh+1-1)*gridSpacing^-1));

correlationMap2D = correlationMap2D(...
    max(1, peakCoordinates(1)-halfNeighborhoodSize):min(end, peakCoordinates(1)+halfNeighborhoodSize),...
    max(1, peakCoordinates(2)-halfNeighborhoodSize):min(end, peakCoordinates(2)+halfNeighborhoodSize));

finerCorrelationMap2D = interp2(meshgridX, meshgridY, correlationMap2D,...
    finerMeshgridX, finerMeshgridY, 'spline');

%% Find new peak of new
% correlation map and calculate pixel coordinate.
[ypeak, xpeak] = find(finerCorrelationMap2D==max(finerCorrelationMap2D(:)));

% Scale back down to pixel units
interpolatedPixelCoordinates = [ypeak xpeak];
interpolatedPixelCoordinates = ...
    floor((interpolatedPixelCoordinates - 1) / gridSpacing^-1) + 1 + ...
    (mod((interpolatedPixelCoordinates - 1), gridSpacing^-1) * gridSpacing);

% Center back around original peak
interpolatedPixelCoordinates = interpolatedPixelCoordinates ...
    - 1 - halfNeighborhoodSize + peakCoordinates;

end
