function StabilizeVideo(inputVideoPath, parametersStructure)
% StabilizeVideo    Generate a stabilized video
%   StabilizeVideo(parametersStructure, fileName) generates a stabilized
%   video by calling MakeMontage. Each time enough strips have accumulated
%   to generate one stabilized video, that frame is written to the video
%   file. We recommend setting stripHeight to a number that divides
%   evenly into the frame height for best results.
%   
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideoInput| is the path to the video.
%
%   See MakeMontage for all relevant parameters of |parametersStructure|
%
%  -----------------------------------
%   Example usage
%   -----------------------------------
%
%   inputVideoPath = 'MyVid.avi';
%   load('MyVid.avi_params.mat');
%   StabilizeVideo(inputVideoPath, parametersStructure);

parametersStructure.stabilizeVideo = 1;

if ~isfield(parametersStructure,'stripHeight')
    parametersStructure.stripHeight = 3;
end

MakeMontage(parametersStructure, inputVideoPath);

end