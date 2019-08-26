function [stripIndices, stripsPerFrame] = DivideIntoStrips(inputVideo, parametersStructure)
%DIVIDE INTO STRIPS Returns coordinates of top left corner of strips.
%   Takes the video input and uses the given parameters
%   to evenly divide the video into strips. It then will return the index
%   in the videoInputArray of the top left corner of each strip and return
%   all of these indices in an array. Each row of the output array
%   represents the indices of a single strip.
%   If the video cannot be evenly divided, then rounding will occur, which
%   have negligible impact on final analysis results.
%
%   Precondition: All inputs have already been validated.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%  samplingRate        :   sampling rate of the video
%  stripHeight         :   the size of each strip

%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
end

%% Set parameters to defaults if not specified.
if ~isfield(parametersStructure, 'samplingRate')
    samplingRate = 540;
    RevasMessage('using default parameter for samplingRate');
else
    samplingRate = parametersStructure.samplingRate;
    if ~IsNaturalNumber(samplingRate)
        error('samplingRate must be a natural number');
    end
end

if ~isfield(parametersStructure, 'stripHeight')
    stripHeight = 15;
    RevasMessage('using default parameter for stripHeight');
else
    stripHeight = parametersStructure.stripHeight;
    if ~IsNaturalNumber(stripHeight)
        error('stripHeight must be a natural number');
    end
end

% Default frame rate if a matrix representation of the video passed in.
% Users may also specify custom frame rate via parametersStructure.
if ~writeResult && ~isfield(parametersStructure, 'FrameRate')
    parametersStructure.FrameRate = 30;
    RevasWarning('using default parameter for FrameRate', parametersStructure);
end

%% Divide into strips.

if writeResult
    reader = VideoReader(inputVideo);
    numberOfFrames = reader.Framerate * reader.Duration;
    height = reader.Height;
    parametersStructure.FrameRate = reader.FrameRate;
else
    [height, ~, numberOfFrames] = size(inputVideo);
end

stripsPerFrame = round(samplingRate / parametersStructure.FrameRate);

stripIndices = zeros(stripsPerFrame*numberOfFrames, 3);

distanceBetweenStrips = (height - stripHeight)...
    / (stripsPerFrame - 1);

% compute the rows of stripIndices
for stripNumber = (1:stripsPerFrame*numberOfFrames)
    
    % Calculate row number and store in the 1st column of output array.
    rowNumber = mod(stripNumber - 1, stripsPerFrame) * distanceBetweenStrips + 1;
    
    % Edge case for if there is only strip per frame.
    if isnan(rowNumber) && stripsPerFrame == 1
        rowNumber = 1;
    end

    % Column number is always 1 since we left align strips.
    columnNumber = 1;
    
    % Calculate frame number and store in the 3rd column of output array.
    frameNumber = (stripNumber-1) / stripsPerFrame + 1;
    
    % Place calculated values into stripIndices
    stripIndices(stripNumber,:) = [rowNumber columnNumber frameNumber];
end

% floor all at once to take advantage of vectorization
stripIndices = floor(stripIndices);
end
