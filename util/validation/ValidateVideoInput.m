function [] = validateVideoInput(videoInput)
%VALIDATE VIDEO INPUT Ensures user input for videos is valid.
%   The function will take no action if the input is valid.
%   An input is valid if it is either a 3D or 4D array.
%   If the input is invalid, an error will be thrown.

numDims = ndims(videoInput);
if numDims ~= 3 && numDims ~= 4
    error('Invalid Input for videoInput (it was not a 3D or 4D array)');

end

