function StabilizeVideo(inputVideoPath, parametersStructure)
% StabilizeVideo    Generate a stabilized video
%   StabilizeVideo(parametersStructure, fileName) generates a stabilized
%   video by calling MakeMontage. Each time enough strips have accumulated
%   to generate one stabilized video, that frame is written to the video
%   file. We recommend setting stripHeight to a number that divides
%   evenly into the frame height for best results.
%   
%   Fields of the |parametersStructure| 
%   -----------------------------------
%   See MakeMontage for all relevant parameters

parametersStructure.stabilizeVideo = 1;
MakeMontage(parametersStructure, inputVideoPath);

end