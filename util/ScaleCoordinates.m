function [newStripPositions] = ScaleCoordinates(stripPositions)
%% ScaleCoordinates   Takes the raw eyePositionTrace output of 
% StripAnalysis and converts it into a usable output for creating the 
% reference frames.

%% Negate all frame positions
newStripPositions = -stripPositions;

%% Scale the strip coordinates so that all values are positive. 
% Take the negative value with the highest magnitude in each column of
% framePositions and add that value to all the values in the column. This
% will zero the most negative value (like taring a weight scale). Then add
% a positive integer to make sure all values are > 0 (i.e., if we leave the
% zero'd value in framePositions, there will be indexing problems later,
% since MatLab indexing starts from 1 not 0).
column1 = newStripPositions(:, 1);
column2 = newStripPositions(:, 2); 

if column1(column1<0)
   mostNegative = max(-1*column1);
   newStripPositions(:, 1) = newStripPositions(:, 1) + mostNegative + 2;
end

if column2(column2<0)
    mostNegative = max(-1*column2);
    newStripPositions(:, 2) = newStripPositions(:, 2) + mostNegative + 2;
end

if column1(column1<0.5)
    newStripPositions(:, 1) = newStripPositions(:, 1) + 2;
end

if column2(column2<0.5)
    newStripPositions(:, 2) = newStripPositions(:, 2) + 2;
end

if any(column1==0)
    newStripPositions(:, 1) = newStripPositions(:, 1) + 2;
end

if any(column2==0)
    newStripPositions(:, 2) = newStripPositions(:, 2) + 2;
end
end