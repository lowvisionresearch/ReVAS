function [stripIndices, stripsPerFrame] = DivideIntoStrips(videoInputPath, parametersStructure)
%DIVIDE INTO STRIPS Returns coordinates of top left corner of strips.
%   Takes the video input in array format and uses the given parameters
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

%% Divide into strips.

reader = VideoReader(videoInputPath);
numberOfFrames = reader.Framerate * reader.Duration;
videoFrameRate = reader.FrameRate;
frame = readFrame(reader);
frameHeight = reader.Height;

stripsPerFrame = round(samplingRate / videoFrameRate);

stripIndices = zeros(stripsPerFrame*numberOfFrames, 3);

distanceBetweenStrips = (frameHeight - stripHeight)...
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
