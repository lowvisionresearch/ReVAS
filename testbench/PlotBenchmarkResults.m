clearvars;
close all
clc;

%% select final mat files or txt computation time files
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
    
    data(i) = load(currentFile,'timeArray','eyePositionTraces');
    [PS_hor(i,:),~] = ComputePowerSpectra(540,data(i).eyePositionTraces(:,1), 1080, .75,0);
    [PS_ver(i,:),f] = ComputePowerSpectra(540,data(i).eyePositionTraces(:,2), 1080, .75,0);
end

[sortedStripHeight,sortI] = sort(stripHeight,'ascend');
data = data(sortI);

%% plot eye positions

cols = jet(length(data));
figure('units','normalized','outerposition',[.5 .5 .5 .3]);
for i=1:length(data)
    subplot(1,2,1)
    plot(data(i).timeArray,data(i).eyePositionTraces(:,1),'-','Color',cols(i,:));
    hold on;
    subplot(1,2,2)
    plot(data(i).timeArray,data(i).eyePositionTraces(:,2),'-','Color',cols(i,:));
    hold on;
    
    legendStr{i} = num2str(sortedStripHeight(i));
end

subplot(1,2,1);
ylabel('Horizontal position');
subplot(1,2,2);
ylabel('Vertical position');
legend(legendStr);

%% plot power spectra
cols = jet(length(data));
figure('units','normalized','outerposition',[.5 .5 .5 .3]);
for i=1:length(data)
    subplot(1,2,1)
    semilogx(f,PS_hor(i,:),'-','Color',cols(i,:));
    hold on;
    subplot(1,2,2)
    semilogx(f,PS_ver(i,:),'-','Color',cols(i,:));
    hold on;
end

subplot(1,2,1);
ylabel('Power Spectra');
xlabel('Temporal Frequency');
title('Horizontal');
subplot(1,2,2);
ylabel('Power Spectra');
xlabel('Temporal Frequency');
legend(legendStr);
title('Vertical');

%% plot computation times
figure('units','normalized','outerposition',[.5 .5 .5 .3]);
for i=1:length(data)

end

ylabel('Computation Time (sec)');
xlabel('Strip Height');
title('Computation Times');
