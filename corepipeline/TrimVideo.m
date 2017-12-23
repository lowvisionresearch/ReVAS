function TrimVideo(inputVideoPath, parametersStructure)
%TRIM VIDEO Removes upper and right edge of video.
%   Removes the upper few rows and right few columns. 
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoPath| is the path to the video. The result is that the
%   trimmed version of this video is stored with '_dwt' appended to the
%   original file name.
%
%   |parametersStructure| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite           : set to true to overwrite existing files.
%                         Set to false to abort the function call if the
%                         files already exist. (default false)
%   borderTrimAmount    : specifies the number of rows and columns to be
%                         removed as a vector with the number of
%                         rows/columns to be removed from each edge
%                         specified in the following order:
%                         [left right top bottom]. The default is
%                         removing 24 from the right and top. If a scalar
%                         is provided instead, then that amount will be
%                         removed from the right and top only.
%                         (default [0 24 24 0])
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = 1;
%       parametersStructure.borderTrimAmount = [0 24 24 0];
%       TrimVideo(inputVideoPath, parametersStructure);

%% Handle overwrite scenarios.
outputVideoPath = [inputVideoPath(1:end-4) '_dwt' inputVideoPath(end-3:end)];
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif nargin == 1 || ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['TrimVideo() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);    
    return;
else
    RevasWarning(['TrimVideo() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);  
end

%% Set parameters to defaults if not specified.

if nargin == 1 || ~isfield(parametersStructure, 'borderTrimAmount')
    borderTrimAmount = [0 24 24 0];
    RevasWarning('using default parameter for borderTrimAmount', parametersStructure);
elseif isscalar(parametersStructure.borderTrimAmount)
    borderTrimAmount = [0 parametersStructure.borderTrimAmount ...
        parametersStructure.borderTrimAmount 0];
else
    borderTrimAmount = parametersStructure.borderTrimAmount;
end
for t = borderTrimAmount
    if ~IsNaturalNumber(t)
        error('borderTrimAmount must consist of natural numbers');
    end
end

%% Trim the video frame by frame

left = borderTrimAmount(1);
right = borderTrimAmount(2);
top = borderTrimAmount(3);
bottom = borderTrimAmount(4);

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

height = size(videoInputArray, 1);
width = size(videoInputArray, 2);
numberOfFrames = size(videoInputArray, 3);

% Preallocate.
trimmedFrames = zeros(height - top - bottom, ...
    width - left - right, numberOfFrames, 'uint8');

for frameNumber = 1:numberOfFrames
    frame = videoInputArray(:,:,frameNumber);
    trimmedFrames(:,:,frameNumber) = ...
        frame(top+1 : height-bottom, ...
       left+1 : width-right);
end

writeVideo(writer, trimmedFrames);
close(writer);

end
