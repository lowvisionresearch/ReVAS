function success = Tester_IsImageFile

imSuccessFile = 'cameraman.tif';
imFailFile = 'nonexistentimage.tif';

if IsImageFile(imSuccessFile) && ...
        ~IsImageFile(imFailFile)
    success = true;
else
    success = false;
end
