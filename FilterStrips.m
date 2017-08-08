function [filteredStripIndices, lengthCutOut, numberOfNaNs] = FilterStrips(stripIndices)
% FilterStrips     Replace all NaNs in a column vector with interpolation
%   FilterStrips(stripIndices) takes a column vector and interpolates
%   between the values "bordering" each strip of consecutive NaNs. Then it
%   returns the new vector.
%   
%   stripIndices should should have dimensions Nx1.
%
%   Example: 
%       A = [1; 2; 3; 4; 5; NaN; NaN; NaN; NaN; 10];
%       x = FilterStrips(A);
%       x(8)
%       ans = 
%            8

% Remove NaNs at the beginning of framePositions
i = 1;
while i < size(stripIndices, 1) && isnan(stripIndices(i))
    i = i + 1;
end
stripIndices(1:i-1,:) = [];
numberOfNaNs = i-1;

% get indices of all NaN values
NaNIndices = find(isnan(stripIndices));
NaNIndicesCopy = NaNIndices;
NaNIndicesStorage = NaNIndices;
lengthCut = 0;

% Every 3 items in startAndEndPairs will be, in this order: last number
% before a strip of consecutive NaNs, first number after a strip of NaNs,
% and number of NaNs in that strip (k). If there is no number after tha NaNs
% (i.e., the NaNs go to the end of the matrix) then there will only be two
% numbers: the last number before the NaNs begin and the number of NaNs
startAndEndPairs = [];
k = 1;

% find the values that "border" the next segment of consecutive NaNs
while ~isempty(NaNIndicesCopy)
    
    dimensions = size(NaNIndicesCopy);
    % If there is one item left in the NaNIndicesCopy, that means there is
    % only one NaN value left. Check if there are more items in the list.
    % If there are, then we proceed with the "bordering" technique we do
    % with the rest of the strips of consecutive NaNs. If not, then just
    % put two values in (last number before NaNs, number of NaNs).
    if dimensions(1) == 1
        lastBeforeNaN = stripIndices(NaNIndices(1) - 1);
        if (NaNIndicesCopy(1) + 1) <= max(size(stripIndices))
            firstAfterNaN = stripIndices(NaNIndicesCopy(1) + 1);
            NaNStrip = [lastBeforeNaN firstAfterNaN k];
            startAndEndPairs = [startAndEndPairs NaNStrip];
            NaNIndicesCopy = [];
        else
            NaNStrip = [lastBeforeNaN k];
            startAndEndPairs = [startAndEndPairs NaNStrip];
            NaNIndicesCopy = [];
        end

    % If the next index in NaNIndicesCopy is one larger than the current
    % index, then those indices are part of the same strip of consecutive
    % NaN values. If that is the case, increment k (number of NaNs in the
    % strip) and move down NaNIndicesCopy to check the next two indices
    elseif NaNIndicesCopy(1) == NaNIndicesCopy(2) - 1 
        k = k + 1;
        NaNIndicesCopy = NaNIndicesCopy(2:end);
        
    % If the next index in NaNIndicesCopy is NOT one larger than the
    % current index, that means that those indices are NOT part of the same
    % strip of consecutive NaNs (i.e., indices are 485, 486, 487, 510,
    % 511). In that case, NaNIndices(1)-1 will be the index of the last
    % number before the NaN strip begins, and NaNIndicesCopy(1) + 1 will be
    % the index of the first number after the NaN strip ends.
    else
        lastBeforeNaN = stripIndices(NaNIndices(1) - 1);
        firstAfterNaN = stripIndices(NaNIndicesCopy(1) + 1);
        NaNStrip = [lastBeforeNaN firstAfterNaN k];
        startAndEndPairs = [startAndEndPairs NaNStrip];
        
        % Continue moving down the matrices to the values we have not
        % checked yet.
        NaNIndices = NaNIndices(k+1:end);
        k = 1;
        NaNIndicesCopy(1) = [];
        %NaNIndicesCopy = NaNIndicesCopy(2:end);
    end

end

% Now that the function has determined the values "bordering" the strips of
% consecutive NaNs, reset the NaNIndices variable so we can insert
% interpolated values into the original stripIndices.
NaNIndices = NaNIndicesStorage;

while ~isempty(startAndEndPairs)
    
    dimensions = size(startAndEndPairs);
    % Remember from earlier that if the NaN strip goes to the end of the
    % matrix, we only put 2 values into startAndEndPairs (last number
    % before NaNs begin and number of NaNs in the strip). In that case,
    % assume the last number before the NaNs remains constant, so set
    % the rest of the NaN values in the matrix equal to that last number.
    if dimensions(2) == 2
        lengthCut = 1;
        lengthCutOut = size(stripIndices(NaNIndices(1):end));
        stripIndices(NaNIndices(1):end) = [];
        startAndEndPairs = [];
        
    % Remember that startAndEndPairs generally has chunks of 3 values: last
    % number before NaNs start, first number after NaNs end, and number
    % of consecutive NaNs in a strip. Therefore, if we assume the change
    % in position between the two real values is linear, then dy is simply 
    % (last number before NaNs - first number after NaNs) / k. Then
    % replace all the NaNs in the original stripIndices with dy
    else
        start = startAndEndPairs(1);
        ending = startAndEndPairs(2);
        numPoints = startAndEndPairs(3) + 1;
        if start == ending
            for point = 1:(numPoints-1)
                stripIndices(NaNIndices(1)) = start;
                NaNIndices(1) = []; 
            end
        else
            dy = (ending-start)/numPoints;
            if max(size(NaNIndices)) == 1
                stripIndices(NaNIndices(1)) = start+dy;
                NaNIndices(1) = [];
            else
                for point = start+dy : dy : ending-dy
                    stripIndices(NaNIndices(1)) = point;
                    NaNIndices(1) = []; 
                end
            end
        end
%         disp([start ending])
%         interpolatedPositions = interp1([start ending], [start ending], start:dy:ending, 'pchip');
%         for point = 2:size(interpolatedPositions,2)-1
%             stripIndices(NaNIndices(1)) = interpolatedPositions(point);
%             NaNIndices = NaNIndices(2:end);
%         end
    end
    startAndEndPairs = startAndEndPairs(4:end);
end

if lengthCut == 0
    lengthCutOut = -1;
end

[filteredStripIndices] = stripIndices;

end