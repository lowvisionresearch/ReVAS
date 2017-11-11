function [] = ValidateReferenceFrame(referenceFrame)
%VALIDATE REFERENCE FRAME Ensures reference frame input is valid.
%   Ensures reference frame input is valid.
%   The function will take no action if the input is invalid.
%   An input is valid if it is a 2D array.
%   If the input is invalid, an error will be thrown.

numDims = ndims(referenceFrame);
if numDims ~= 2
    error('Invalid Input for referenceFrame (it was not a 2D array)');

end

