function [result] = Crop(image)
%% Crop     Crop out 0 padding from an image
%   Crop(image, width, height, margin) takes in a two-dimensional matrix, its 
%   width, its height, and a margin size for which it tests for 0's. The
%   function removes 0 padding, given that there is a region of size 'margin' that
%   is made up entirely of 0's. 
%
%   Example: 
%       A = [1 2 3 4 5 0 0; 
%            1 2 3 4 5 0 0; 
%            0 0 0 0 0 0 0; 
%            0 0 0 0 0 0 0]
%       x = Crop(A, 7, 4, 2)
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