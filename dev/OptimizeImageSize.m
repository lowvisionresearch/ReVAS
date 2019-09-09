function [outIm, newM, newN] = OptimizeImageSize(im)

[m,n] = size(im);
change = 2.^round(log2([m n])) - [m n];
outIm = im;

% rows
if change(1)<0
    outIm(1:abs(change(1)),:) = [];
elseif change(1)>0
    outIm = [outIm; zeros(change(1),n)];
end


% columns
if change(2)<0
    outIm(:,1:abs(change(2))) = [];
elseif change(2)>0
    outIm = [outIm zeros(size(outIm,1),change(2)) ];
end

[newM, newN] = size(outIm);
    


