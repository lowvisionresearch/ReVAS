function [videoInputArray, videoFrameRate] = VideoPathToArray(videoInputPath)
%VIDEO PATH TO ARRAY Takes the path to a video file and converts it to an
%array.
%   Takes the path to a video file and converts it to either a 3D or 4D
%   array, depending if there are multiple color channels.
%   Also returns the frame rate of the video.
%   If the path input is invalid, an error will be raised.

if ischar(videoInputPath)    
    % Determine dimensions needed for purposes of preallocation.
    reader = VideoReader(videoInputPath);
    numberOfFramesOfVideoInput = reader.NumberOfFrames;
    videoFrameRate = reader.FrameRate;
    videoWidth = reader.Width;
    videoHeight = reader.Height;
    videoInputArray = zeros(videoHeight, videoWidth, numberOfFramesOfVideoInput, 'uint8');
    
    % Remake this variable since readFrame() cannot be called after
    % NumberOfFrames property is accessed.
    reader = VideoReader(videoInputPath);
    
    % Fill the output array with all of the frames.
    for k = (1:numberOfFramesOfVideoInput)
        frame = double(readFrame(reader))/255;
        if ndims(frame) == 3
            frame = rgb2gray(frame);
        end
        videoInputArray(:,:,k) = frame;
    end
        
else
    error('Invalid Input for videoPathToArray (it was not a char array)');
end

end

