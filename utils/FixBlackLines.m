function newFrame = FixBlackLines(frame)
% newFrame = FixBlackLines(frame)
%
%   Looks for black rows in a grayscale image and fixes them with linear
%   interpolation
%
%   'frame' is a 2D array.
%
% MNA 2/21/2020


% find black lines
blackRows = sum(frame,2) == 0;

% exclude black boundaries from filling mask
blackRows(1:find(~blackRows,1,'first')) = false;
blackRows(find(~blackRows,1,'last'):end) = false;

% get a mask to do the filling
mask = repmat(blackRows,1,size(frame,2));

newFrame = regionfill(frame,mask);

