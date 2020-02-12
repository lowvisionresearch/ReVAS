function videoArray = ReadVideoToArray(inputVideoPath)
% videoArray = ReadVideoToArray(inputVideoPath)
%
%   A function to read video files into a 3D array. Takes in full path to
%   video and outputs the videoArray.
%
%
%  MNA 11/29/19 wrote the initial version

if ~exist(inputVideoPath,'file')
    videoArray = []; %#ok<NASGU>
    error('Video file not found.');
end

reader = VideoReader(inputVideoPath);
numberOfFrames = reader.Framerate * reader.Duration;

% preallocate
videoArray = zeros(reader.Height, reader.Width, numberOfFrames, 'uint8');

% read frames
for frameNumber = 1:numberOfFrames
	frame = readFrame(reader);
	[~, ~, numChannels] = size(frame);
    
    % handle RGB input (if any for a weird reason!)
	if numChannels == 1
		videoArray(1:end, 1:end, frameNumber) = frame;
	else
		videoArray(1:end, 1:end, frameNumber) = rgb2gray(frame);
	end
end