function frame = ReadSpecificFrame(readerObject,frameNumber)
% frame = ReadSpecificFrame(videoPath,frameNumber)
%
%

if nargin<1 || ~isobject(readerObject)
    error('ReadSpecificFrame: readerObject must be a video reader object');
end

if nargin<2 || isempty(frameNumber) || ~IsPositiveInteger(frameNumber)
    error('ReadSpecificFrame: frameNumber must be a positive integer');
end

info = get(readerObject);
readerObject.CurrentTime = (frameNumber-1)/info.FrameRate;
frame = readFrame(readerObject);

% handle rgb
if size(frame,3)>1
    frame = rgb2gray(frame);
end

