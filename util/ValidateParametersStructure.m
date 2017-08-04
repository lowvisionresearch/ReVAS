function [] = validateParametersStructure(parametersStructure)
%VALIDATE PARAMETERS STRUCTURE Ensures parameters structure input is valid.
%   Ensures parameters structure input is valid.
%   The function will take no action if the input is valid.
%   An input is valid if ...
%   If the input is invalid, an error will be thrown.

% stripHeight and stripWidth
if ~isfield(parametersStructure, 'stripHeight')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a stripHeight field)');
end
if ~isfield(parametersStructure, 'stripWidth')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a stripWidth field)');
end

% samplingRate
if ~isfield(parametersStructure, 'samplingRate')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a samplingRate field)');
end

% enableSubpixelInterpolation (enables/disables interpolation)
if ~isfield(parametersStructure, 'enableSubpixelInterpolation')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a enableSubpixelInterpolation field)');
end

% subpixelInterpolationParameters
if parametersStructure.enableSubpixelInterpolation && ...
        ~isfield(parametersStructure, 'subpixelInterpolationParameters')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a subpixelInterpolationParameters field)');
end

% adaptiveSearch (enables/disables confined/adaptive search for
% cross-correlation peak)
if ~isfield(parametersStructure, 'adaptiveSearch')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a adaptiveSearch field)');
end

% minimumPeakRatio
if ~isfield(parametersStructure, 'minimumPeakRatio')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a minimumPeakRatio field)');
end

% enableVerbosity
if ~isfield(parametersStructure, 'enableVerbosity')
    error('Invalid Input for validateParametersStructure (parametersStructure does not have a enableVerbosity field)');
end
