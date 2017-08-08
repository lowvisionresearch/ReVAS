clearvars;
% close all
clc;

addpath(genpath('..'));

%% select txt computation time files
filenames = uipickfiles;
if ~iscell(filenames)
    if filenames == 0
        fprintf('User cancelled file selection. Silently exiting...\n');
        return;
    end
end

for i=1:length(filenames)
    currentFile = filenames{i};
    
    heightStart = strfind(currentFile,'STRIPHEIGHT-');
    heightEnd = strfind(currentFile(heightStart:end),'_');
    stripHeight(i) = str2double(currentFile(heightStart+length('STRIPHEIGHT-'):heightStart+heightEnd-2));
    
    timeStart = strfind(currentFile,'TIMEELAPSED-');
    timeEnd = strfind(currentFile(timeStart:end),'.');
    times(i) = str2double(currentFile(timeStart+length('TIMEELAPSED-'):timeStart+timeEnd-2));
    
end

[sortedStripHeight,sortI] = sort(stripHeight,'ascend');
times = times(sortI);

%% plot computation times
figure('units','normalized','outerposition',[.5 .5 .5 .3]);
plot(sortedStripHeight, times);
ylabel('Computation Time (sec)');
xlabel('Strip Height');
title('Computation Times');
