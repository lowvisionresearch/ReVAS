function StabilizeVideo(inputVideoPath, parametersStructure)
% StabilizeVideo    Generate a stabilized video
%   StabilizeVideo(parametersStructure, fileName) generates a stabilized
%   video by calling MakeMontage. Each time enough strips have accumulated
%   to generate one stabilized video, that frame is written to the video
%   file.
%   
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   See MakeMontage for all relevant parameters

parametersStructure.stabilizeVideo = 1;
MakeMontage(parametersStructure, inputVideoPath);

end