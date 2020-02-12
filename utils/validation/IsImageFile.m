function result = IsImageFile(value)
%IS IMAGE FILE Checks if value is a path to an image file.
%   Checks if value is a path to an image file. Returns true if so, else false.

try
    imfinfo(value);
catch
    result = false;
    return;
end

result = true;
    
end