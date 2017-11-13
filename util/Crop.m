function [result] = Crop(image)
%% Crop     Crop out 0 padding from an image
%   Crop takes a 2D array and removes all zero padding. Before doing so,
%   all NaNs are converted to zeros.
%
%   Example: 
%       A = [1 2 3 4 5 0 0; 
%            1 2 3 4 5 0 0; 
%            0 0 0 0 0 NaN 0; 
%            0 0 0 0 0 0 NaN]
%       x = Crop(A)
%       x = 
%           1 2 3 4 5
%           1 2 3 4 5

%% Convert any NaN values in the reference frame to a 0.
indices = isnan(image);
image(indices) = 0;

% Crop out the leftover 0 padding from the original template. First check
% for 0 columns
indices = image == 0;
sumColumns = sum(indices);
columnsToRemove = sumColumns == size(image, 1);
image(:, columnsToRemove) = [];

% Then check for 0 rows
indices = image == 0;
sumRows = sum(indices, 2);
rowsToRemove = sumRows == size(image, 2);
image(rowsToRemove, :) = [];

result = image;
end