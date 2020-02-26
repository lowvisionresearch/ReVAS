function stimulus = MakeStimulusCross(crossSize, crossThickness, polarity)

if nargin < 1 || isempty(crossSize)
    crossSize = 11; % pixels
end

if nargin < 2 || isempty(crossThickness)
    crossThickness = 1; % pixels
end

if nargin < 3 || isempty(polarity)
    polarity = 1;
end

% Both size and thickness must be odd
if mod(round(crossSize), 2) == 0
    crossSize = round(crossSize) + 1;
end
if mod(round(crossThickness), 2) == 0
    crossThickness = round(crossThickness) + 1;
end
if crossSize < crossThickness
    error('cross size must be greater than its thickness');
end

% Draw the stimulus cross based on struct parameters provided.
padN = (crossSize - crossThickness)/2;

hor = repmat([false(padN,1); true(crossThickness,1); false(padN,1)],1,crossSize);
ver = repmat([false(1,padN) true(1,crossThickness) false(1,padN)],crossSize,1);
stimulus = hor | ver;

if ~polarity
    stimulus = ~stimulus;
end



