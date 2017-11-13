function TrimVideo(inputVideoPath, parametersStructure)
%TRIM VIDEO Removes upper and right edge of video
%   Removes the upper few rows and right few columns. 
%
%   |inputVideoPath| is the path to the video. The result is that the
%   trimmed version of this video is stored with '_dwt' appended to the
%   original file name. 
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite           :   set to 1 to overwrite existing files resulting 
%                           from calling the function. Set to 0 to abort 
%                           the function call if the files exist in the 
%                           current directory.
%   borderTrimAmount    :   specifies the number of rows and columns to be
%                           removed. Default value is 24 pixels if no value 
%                           is specified by the user.
%
%   Example usage: 
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = 1;
%       parametersStructure.borderTrimAmount = 24;
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
    borderTrimAmount = 24;
    RevasWarning('using default parameter for borderTrimAmount', parametersStructure);
else
    borderTrimAmount = parametersStructure.borderTrimAmount;
    if ~IsNaturalNumber(borderTrimAmount)
        error('borderTrimAmount must be a natural number');
    end
end

%% Trim the video frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

height = size(videoInputArray, 1);
width = size(videoInputArray, 2);
numberOfFrames = size(videoInputArray, 3);

% Preallocate.
trimmedFrames = zeros(height - borderTrimAmount, ...
    width - borderTrimAmount, numberOfFrames, 'uint8');

for frameNumber = 1:numberOfFrames
    frame = videoInputArray(:,:,frameNumber);
    trimmedFrames(:,:,frameNumber) = ...
        frame(borderTrimAmount+1 : height, ...
       1 : width-borderTrimAmount);
end

writeVideo(writer, trimmedFrames);
close(writer);

end
