clearvars;
close all
clc;

addpath(genpath('..'));

%% select final mat files
for j=1:1

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
        allUnderScores = strfind(currentFile(heightStart:end),'_');
        heightEnd = allUnderScores(1);
        stripHeight(i) = str2double(currentFile(heightStart+length('STRIPHEIGHT-'):heightStart+heightEnd-2));

        samplingRateStart = strfind(currentFile,'SAMPLINGRATE-');
        if isempty(samplingRateStart)
            samplingRate(i) = 540;
        else
            allUnderScores = strfind(currentFile(samplingRateStart:end),'_');
            samplingRateEnd = allUnderScores(1);
            samplingRate(i) = str2double(currentFile(samplingRateStart+length('SAMPLINGRATE-'):samplingRateStart+samplingRateEnd-2));
        end

        data(i) = load(currentFile,'timeArray','eyePositionTraces');
        [PS_hor{i},~] = ComputePowerSpectra(samplingRate(i),data(i).eyePositionTraces(:,1), samplingRate(i)*2, .75,0);
        [PS_ver{i},f{i}] = ComputePowerSpectra(samplingRate(i),data(i).eyePositionTraces(:,2), samplingRate(i)*2, .75,0);
    end

    [sortedStripHeight,sortI] = sort(stripHeight,'ascend');
    data = data(sortI);

    %% plot eye positions
    cols = jet(length(data));
    if j==1
        fh1 = figure('units','normalized','outerposition',[.5 .5 .5 .3]);
    else
        figure(fh1);
    end
    for i=1:length(data)
        subplot(1,2,1)
        plot(data(i).timeArray,data(i).eyePositionTraces(:,1),'-','Color',cols(i,:),'LineWidth',j);
        hold on;
        subplot(1,2,2)
        plot(data(i).timeArray,data(i).eyePositionTraces(:,2),'-','Color',cols(i,:),'LineWidth',j);
        hold on;

        legendStr{i} = num2str(sortedStripHeight(i));
    end

    subplot(1,2,1);
    ylabel('Horizontal position');
    subplot(1,2,2);
    ylabel('Vertical position');
    if j==1
        legend(legendStr);
    end

    %% plot power spectra
    cols = parula(length(data));
    if j==1
        fh2 = figure('units','normalized','outerposition',[.5 .5 .5 .3]);
    else
        figure(fh2);
    end
    for i=1:length(data)
        subplot(1,2,1)
        semilogx(f{sortI(i)},PS_hor{sortI(i)},'-','Color',cols(i,:),'LineWidth',j);
        hold on;
        subplot(1,2,2)
        semilogx(f{sortI(i)},PS_ver{sortI(i)},'-','Color',cols(i,:),'LineWidth',j);
        hold on;
    end

    subplot(1,2,1);
    ylabel('Power Spectra');
    xlabel('Temporal Frequency');
    title('Horizontal');
    subplot(1,2,2);
    ylabel('Power Spectra');
    xlabel('Temporal Frequency');
    if j==1
        legend(legendStr);
    end
    title('Vertical');

end
