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

%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Trim the video frame by frame

left = borderTrimAmount(1);
right = borderTrimAmount(2);
top = borderTrimAmount(3);
bottom = borderTrimAmount(4);

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

% Determine dimensions of video.
reader = VideoReader(inputVideoPath);
numberOfFrames = reader.NumberOfFrames;
width = reader.Width;
height = reader.Height;

% Remake this variable since readFrame() cannot be called after
% NumberOfFrames property is accessed.
reader = VideoReader(inputVideoPath);

% Read, trim, and write frame by frame.
for frameNumber = 1:numberOfFrames
    if ~abortTriggered
        frame = readFrame(reader);
        frame = frame(top+1 : height-bottom, ...
           left+1 : width-right);
       writeVideo(writer, frame);
    end
end

close(writer);

end
