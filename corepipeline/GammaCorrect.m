function GammaCorrect(inputVideoPath, parametersStructure)
%GAMMA CORRECT Applies gamma correction to the video
%   The result is stored with '_gamscaled' appended to the input video file
%   name.
%
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   overwrite          :        set to 1 to overwrite existing files resulting 
%                               from calling the function.
%                               Set to 0 to abort the function call if the
%                               files exist in the current directory.
%   gammaExponent      :        gamma specifies the shape of the curve 
%                               describing the relationship between the 
%                               values in I and J, where new intensity
%                               values are being mapped from I (a frame) 
%                               to J. gammaExponent is a scalar value.
%                               Defaults to 0.6 if no value is specified.
%                             
%   Example usage: 
%       inputVideoPath = 'MyVid.avi';
%       parametersStructure.overwrite = 1;
%       parametersStructure.gammaExponent = 0.6;
%       GammaCorrect(inputVideoPath, parametersStructure);

%% Handle overwrite scenarios.
outputVideoPath = [inputVideoPath(1:end-4) '_gamscaled' inputVideoPath(end-3:end)];
if ~exist(outputVideoPath, 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    RevasWarning(['GammaCorrect() did not execute because it would overwrite existing file. (' outputVideoPath ')'], parametersStructure);
    return;
else
    RevasWarning(['GammaCorrect() is proceeding and overwriting an existing file. (' outputVideoPath ')'], parametersStructure);
end

%% Set parameters to defaults if not specified.

if ~isfield(parametersStructure, 'gammaExponent')
    gammaExponent = 0.6;
    RevasWarning('using default parameter for gammaExponent', parametersStructure);
else
    gammaExponent = parametersStructure.gammaExponent;
    if ~IsRealNumber(gammaExponent)
       error('gammaExponent must be a real number'); 
    end
end

%% Gamma correct frame by frame

writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
open(writer);

[videoInputArray, ~] = VideoPathToArray(inputVideoPath);

numberOfFrames = size(videoInputArray, 3);

for frameNumber = 1:numberOfFrames
    
    videoInputArray(:,:,frameNumber) = ...
        imadjust(videoInputArray(:,:,frameNumber), [], [], gammaExponent);
    
end

writeVideo(writer, videoInputArray);
close(writer);

end

