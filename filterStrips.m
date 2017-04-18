function [filteredStripIndices] = filterStrips(stripIndices)

% get indices of all NaN values
indices = find(isnan(stripIndices));
indicesCopy = indices;

startAndEndPairs = [];
k = 1;

% find the values that "border" the next segment of consecutive NaNs
while ~isempty(indicesCopy)
    dimensions = size(indicesCopy);
    if dimensions(1) == 1
        lastBeforeNaN = stripIndices(indices(1) - 1);
        NaNStrip = [lastBeforeNaN k];
        startAndEndPairs = [startAndEndPairs NaNStrip];
        indicesCopy = [];
    elseif indicesCopy(1) == indicesCopy(2) - 1 
        k = k + 1;
        indicesCopy = indicesCopy(2:end);
    else
        lastBeforeNaN = stripIndices(indices(1) - 1);
        firstAfterNaN = stripIndices(indicesCopy(1) + 1);
        NaNStrip = [lastBeforeNaN firstAfterNaN k];
        startAndEndPairs = [startAndEndPairs NaNStrip];
        indices = indices(k+1:end);
        k = 1;
        indicesCopy = indicesCopy(2:end);
    end

end

% reset the indices variable
indices = find(isnan(stripIndices));

while ~isempty(startAndEndPairs)
    dimensions = size(startAndEndPairs);
    if dimensions(2) == 2
        stripIndices(indices(1):end) = startAndEndPairs(1);
        startAndEndPairs = [];
    else
        start = round(startAndEndPairs(1));
        ending = round(startAndEndPairs(2));
        numPoints = startAndEndPairs(3) + 1;
        dy = (ending-start)/numPoints;
        for point = start+dy:dy:ending
            stripIndices(indices(1)) = point;
            indices = indices(2:end);
        end
    end
    startAndEndPairs = startAndEndPairs(4:end);
end


[filteredStripIndices] = round(stripIndices);

end