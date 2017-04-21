function [videoInputArray, videoFrameRate] = VideoPathToArray(videoInputPath)
%VIDEO PATH TO ARRAY Takes the path to a video file and converts it to an
%array.
%   Takes the path to a video file and converts it to either a 3D or 4D
%   array, depending if there are multiple color channels.
%   Also returns the frame rate of the video.
%   If the path input is invalid, an error will be raised.

if ischar(videoInputPath)    
    % Determine dimensions needed for purposes of preallocation.
    videoReaderOfInput = VideoReader(videoInputPath);
    numberOfFramesOfVideoInput = videoReaderOfInput.NumberOfFrames;
    videoFrameRate = videoReaderOfInput.FrameRate;
    videoWidth = videoReaderOfInput.Width;
    videoHeight = videoReaderOfInput.Height;
    videoInputArray = zeros(videoWidth, videoHeight, numberOfFramesOfVideoInput, 'uint8');
    
    % Remake this variable since readFrame() cannot be called after
    % NumberOfFrames property is accessed.
    videoReaderOfInput = VideoReader(videoInputPath);
    
    % Fill the output array with all of the frames.
    for k = (1:numberOfFramesOfVideoInput)
        videoInputArray(:,:,k) = readFrame(videoReaderOfInput);
    end
        
else
    error('Invalid Input for videoPathToArray (it was not a char array)');
end

end

