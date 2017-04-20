function [result] = Crop(image, width, height, margin)
% Crop     Crop out 0 padding from an image
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

% Get row/column vectors made up of logicals. (1 means there is at least one
% non-zero value within a row or column. 0 means the entire row or column is 
% made up of 0's)
rowBoolean = any(image, 2);
columnBoolean = any(image, 1);

rowDimensions = size(rowBoolean);
columnDimensions = size(columnBoolean);

for row = 1:rowDimensions(1) 
    % checking that row+margin won't exceed matrix dimensions
    if row+margin <= size(rowBoolean(1))
        % if the margin space is entirely 0's, then save the row number
        if rowBoolean(row:row+margin, 1) == 0
            break
        end
    else
        % if row+margin exceeds matrix dimensions, just check if row to the
        % end of the matrix is made up entirely of 0's
        if rowBoolean(row:end, 1) == 0
            break
        end
    end
end

% Same idea for columns, from row processing
for column = 1:columnDimensions(2) 
    if column+margin <= size(columnBoolean(2))
        if columnBoolean(1, column:column+margin) == 0
            break
        end
    else
        if columnBoolean(1, column:end) == 0
            break
        end
    end
end

% Use saved row/column value as the lower limit and the edge of the matrix
% as the upper limit to specify the area to be sliced (cropped) out.

image(row:height, :) = [];
image(:, column:width) = [];

[result] = image;
end