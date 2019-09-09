%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example tests C++ MEX-file matchTemplateOCV_GPU. The MEX function
% uses cv::gpu::matchTemplate from OpenCV to search for matches between an
% image patch and an input image on the GPU.
%
% Copyright 2014 The MathWorks, Inc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup
% Set up test data
peppers   = rgb2gray(imread('peppers.png'));
rectOnion = [191    80    77    55];
template  = imcrop(peppers,rectOnion);

% Copy data to the Gpu
peppers  = gpuArray(peppers);
template = gpuArray(template);

%% Run template matching on the GPU
% Invoke mex function to search for matches between an image patch and an
% input image.
result =  matchTemplateOCV_GPU(template, peppers);

%% Show Results
% Show the input image and the result of normalized cross correlation
subplot(1,2,1);
imshow(peppers); title('Input Image');
subplot(1,2,2);
imshow(result,[]); title('Result of running matchTemplateOCV_GPU()');
truesize; % make the figure tight

% Compute the peak location on the GPU.
[~, idx] = max(abs(result(:)));
[y, x] = ind2sub(size(result),idx(1));

hold('on'); plot(x,y,'ro');

% Plot approximate outline of the onion template
bbox = [x y rectOnion(3:4)] - [round(rectOnion(3:4)/2) 0 0];
bbox = gather(bbox); % bring bbox to the CPU for plotting

rectangle('Position', bbox,'EdgeColor',[1 0 0]);

% Show the template image in a separate window
figure; imshow(template); title('Template');
