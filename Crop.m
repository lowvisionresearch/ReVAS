function [result] = Crop(image)
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

row = 1;
while row <= size(image, 1)
    if image(row, :) == 0
        image(row, :) = [];
        continue
    elseif isnan(image(row, :)) == 1
        image(row, :) = [];
        continue
    end
    row = row + 1;
end

column = 1;
while column <= size(image, 2)
    if image(:, column) == 0
        image(:, column) = [];
        continue
     elseif isnan(image(:, column)) == 1
         image(:, column) = [];
         continue
    end
    column = column + 1;
end

[result] = image;

end