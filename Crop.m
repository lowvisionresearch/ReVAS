function [result] = Crop(image, width, height)

rowBoolean = any(image, 2);
columnBoolean = any(image, 1);

rowDimensions = size(rowBoolean);
columnDimensions = size(columnBoolean);

% grab the first row that is entirely 0
for row = 1:rowDimensions(1) 
    if rowBoolean(row, 1) == 0
        break
    end
end

% grab the first column that is entirely 0
for column = 1:columnDimensions(2) 
    if columnBoolean(1, column) == 0
        break
    end
end

image(row:height, :) = [];
image(:, column:width) = [];

[result] = image;
end