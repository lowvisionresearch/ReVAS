function params = HandleBadFrames(numberOfFrames, params, callerStr)

if nargin < 3 
    callerStr = '';
end

% if badFrames is provided and has the right length
if length(params.badFrames) == numberOfFrames
    params.skipFrame = params.badFrames;
end

% If badFrames is not provided, use all frames
if length(params.badFrames)<numberOfFrames 
    params.skipFrame = false(numberOfFrames,1);
    params.badFrames = false(numberOfFrames,1);
end

% If badFrames are provided but its size don't match the number of frames
if length(params.badFrames) > numberOfFrames
    if (length(params.badFrames) - numberOfFrames) ~= sum(params.badFrames)
        error('HandleBadFrames: invalid badFrames array.');
    end
    params.skipFrame = false(numberOfFrames,1);
    RevasWarning([callerStr ': size mismatch between ''badFrames'' ' ...
        'and input video. Using all frames for this video.'], params);  
end