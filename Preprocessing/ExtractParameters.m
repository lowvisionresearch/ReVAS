% Takes in a trial matrix from PreProcess as well as a patient's
% corresponding patientID and uses this information to create and
% return a struct containing the necessary parameters for a particular
% trial.

function parametersStructure = ExtractParameters(trial, patientID)
    field1 = 'trialNumber'; value1 = trial(1);
    field2 = 'xPosition'; value2 = trial(2);
    field3 = 'yPosition'; value3 = trial(3);
    field4 = 'numLetters'; value4 = trial(4);
    field5 = 'letterSize'; value5 = trial(5);
    field6 = 'videoName'; value6 = strcat(patientID,'_',num2str(value1));
    
    parametersStructure = struct(field1,value1, field2,value2, field3,value3, field4,value4, field5,value5, field6,value6);