function trials = PreProcess(textFile, numTrials, patientID)
% Takes in a text file which contains the information from a patient's
% corresponding .psy file and extracts the relevant parameters from this
% file for each trial.
% 
% textFile: a text file that contains all relevant information from the
% psy file of a given patient.
%
% numTrials: the number of trials for a given patient.
% 
% patientID: the ID corresponding to a particular patient. This is the
% prefix of all the video names for a patient.
%
% numParams: the number of parameters for each trial. For example,
% xPosition, yPosition, and videoName would correspond to a numParams value
% of 3.
%
% Returns an array of trial structs that contain the following parameters for a 
% given trial: trialNumber, xPosition, yPosition, numLetters, letterSize, and videoName.

%% Initialize array of structs that will be returned.
trials(numTrials) = struct('trialNumber',[],'xPosition',[],'yPosition',[], 'numLetters',[], 'letterSize',[],'videoName',[]);


%% Converts contents of textFile into an array of floating point numbers.
formatSpec = '%f';
fileID = fopen(textFile, 'r');
A = fscanf(fileID, formatSpec);
for trialNumber = 0 : numTrials - 1
    currentIndex = trialNumber * 5 + 1;
    trial = A(currentIndex: currentIndex + 5 - 1);
    trials(trialNumber + 1) = ExtractParameters(trial, patientID);
end
