%BENCHMARKING APPLY NOISE Script to apply artificial noise to videos.
%   Script to apply artificial noise to videos.

clc;
clear;
close all;

addpath(genpath('..'));

benchmarkingVideos = cell(1, 7);
benchmarkingVideos{1} = 'testbench\benchmark\benchmark_noise\horizontal_1.avi';
benchmarkingVideos{2} = 'testbench\benchmark\benchmark_noise\horizontal_2.avi';
benchmarkingVideos{3} = 'testbench\benchmark\benchmark_noise\jerky.avi';
benchmarkingVideos{4} = 'testbench\benchmark\benchmark_noise\static.avi';
benchmarkingVideos{5} = 'testbench\benchmark\benchmark_noise\vertical_1.avi';
benchmarkingVideos{6} = 'testbench\benchmark\benchmark_noise\vertical_2.avi';
benchmarkingVideos{7} = 'testbench\benchmark\benchmark_noise\wobble.avi';

noiseAmounts = cell(1,5);
noiseAmounts{1} = '0100';
noiseAmounts{2} = '0266';
noiseAmounts{3} = '0707';
noiseAmounts{4} = '1880';
noiseAmounts{5} = '5000';


parfor i = 1:7
    % Grab path out of cell.
    videoPath = benchmarkingVideos{i};
    
    for j = 1:5
        noise = noiseAmounts{j};
        
        outputVideoPath = [videoPath(1:end-4) '_NOISE-' noise videoPath(end-3:end)];

        writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
        open(writer);

        [videoInputArray, ~] = VideoPathToArray(videoPath);

        height = size(videoInputArray, 1);
        width = size(videoInputArray, 2);
        numberOfFrames = size(videoInputArray, 3);

        % Preallocate.
        noisyFrames = zeros(height, width, numberOfFrames, 'uint8');

        % Add noise frame by frame.
        for frameNumber = 1:numberOfFrames
            frame = videoInputArray(:,:,frameNumber);
            noisyFrames(:,:,frameNumber) = imnoise(frame,'gaussian',str2double(['0.' noise]));
        end

        writeVideo(writer, noisyFrames);
        close(writer);
    end

end

fprintf('Process Completed\n');
