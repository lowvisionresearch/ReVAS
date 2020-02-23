function InstallRevas

% add paths
if isunix
    separator = ':';
else
    separator = ';';
end
paths = regexp(genpath(pwd),separator,'split');

% remove git folders
paths(contains(paths,'git')) = [];

% remove the last empty cell
paths(end) = [];

% make it a single char array
p = [];
for i=1:length(paths)
    p = [p paths{i} separator];
end

% add and save the path
addpath(p);
savepath;

%% compile template matching functions from source codes
% openCV template matching CPU
cd('third_party/visionopencv/TemplateMatching');
fprintf('\nAttempting to compile CPU version:\n'); 
try 
    mexOpenCV matchTemplateOCV.cpp; 
catch
    disp('matchTemplateOCV compilation failed.'); 
end

% openCV template matching GPU
cd('../TemplateMatchingGPU'); 
try
    if isunix
        fprintf('\nAttempting to compile GPU Linux/Mac version:\n');
         mexOpenCV matchTemplateOCV_GPU.cpp -lmwgpu -lmwocvgpumex -largeArrayDims; 
    else
        fprintf('\nAttempting to compile GPU PC version:\n'); 
        mexOpenCV matchTemplateOCV_GPU.cpp -lgpu -lmwocvgpumex -largeArrayDims; 
    end
catch
    disp('matchTemplateOCV_GPU compilation failed.')
end

% template matching using CUDA
cd('../../cuda'); 
fprintf('\nAttempting to compile CUDA implementation.:\n'); 
try 
    mexcuda -lcufft cuda_match.cpp helper/convolutionFFT2D.cu helper/cuda_utils.cu; 
catch
    disp('Compilation of CUDA code failed.'); 
end
cd('../../')
