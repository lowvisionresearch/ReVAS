function ORBOpenCVBuildInfo(buildInfo, context)
% The function adds platform dependent libraries to the build info
% Copyright 2017 The MathWorks, Inc.

% File extensions
%   for windows: linkLibExt = '.lib', execLibExt = '.dll'
[~, linkLibExt, execLibExt] = context.getStdLibInfo();
group = 'BlockModules';

% Platform specific link and non-build files
arch            = computer('arch');
pathBinArch     = fullfile(matlabroot,'bin',arch,filesep);

%--------------------------------------------------------------------------
% Set OpenCV version
%--------------------------------------------------------------------------
ocv_version = '3.4.0';

switch arch
    case {'win32','win64'}
        % Include path to OpenCV .lib files
        linkLibPath = fullfile(matlabroot,'toolbox', ...
            'vision','builtins','src','ocvcg', 'opencv', arch, 'lib');

        ocv_ver_no_dots = strrep(ocv_version,'.','');

        % Associate OpenCV libraries
        nonBuildFilesNoExt = {};
        ocvNonBuildFilesNoExt = AddDefaultOpenCVLibraries( ...
            nonBuildFilesNoExt, ocv_ver_no_dots);
        ocvNonBuildFilesNoExt = AddFeaturesLib( ...
            ocvNonBuildFilesNoExt, ocv_ver_no_dots);
        ocvNonBuildFilesNoExt = AddFlannLib( ...
            ocvNonBuildFilesNoExt, ocv_ver_no_dots);

        ocvLinkFilesNoExt = ocvNonBuildFilesNoExt;
        nonBuildFilesNoExt = [ocvNonBuildFilesNoExt, 'tbb'];
        nonBuildFiles = strcat(pathBinArch,nonBuildFilesNoExt, execLibExt);

        linkFiles = strcat(ocvLinkFilesNoExt, linkLibExt);

    case {'glnxa64','maci64'}
        linkLibPath = pathBinArch;

        ocv_major_ver = ocv_version(1:end-2);
        %
        ocvNonBuildFilesNoExt = { ...
            'libopencv_calib3d', ...
            'libopencv_core', ... 
            'libopencv_features2d', ...
            'libopencv_flann', ...  
            'libopencv_imgproc', ...
            'libopencv_ml', ...
            'libopencv_objdetect', ...
            'libopencv_video', ...
            'libopencv_cudaarithm', ... 
            'libopencv_cudabgsegm', ... 
            'libopencv_cudafeatures2d', ... 
            'libopencv_cudafilters', ... 
            'libopencv_cudaimgproc', ... 
            'libopencv_cudalegacy', ... 
            'libopencv_cudaobjdetect', ... 
            'libopencv_cudaoptflow', ... 
            'libopencv_cudastereo', ... 
            'libopencv_cudawarping', .... 
            'libopencv_cudev', ...
            };

        if strcmpi(arch,'glnxa64')
            nonBuildFiles = strcat(pathBinArch,ocvNonBuildFilesNoExt, ...
                strcat('.so.',ocv_major_ver));

            % Add link files
            linkFiles = strcat(ocvNonBuildFilesNoExt, ...
                strcat('.so.',ocv_major_ver));

            % TBB
            nonBuildFiles = AddTbbLibs(nonBuildFiles, pathBinArch);

            % Glnxa64 specific runtime libraries
            nonBuildFiles = AddGLNXRTlibs(nonBuildFiles);
        else % MAC
            nonBuildFiles = strcat(pathBinArch,ocvNonBuildFilesNoExt, ...
                strcat('.',ocv_major_ver,'.dylib'));

            % Add link files
            linkFiles = strcat(ocvNonBuildFilesNoExt, ...
                strcat('.',ocv_major_ver,'.dylib'));

            % TBB
            nonBuildFiles = AddTbbLibs(nonBuildFiles, pathBinArch);
        end

    otherwise
        % unsupported
        assert(false,[ arch ' operating system not supported']);
end

nonBuildFiles = AddCUDALibs(nonBuildFiles, pathBinArch);

linkPriority    = '';
linkPrecompiled = true;
linkLinkonly    = true;
buildInfo.addLinkObjects(linkFiles,linkLibPath,linkPriority,...
    linkPrecompiled,linkLinkonly,group);

buildInfo.addNonBuildFiles(nonBuildFiles,'',group);

%==========================================================================
function nonBuildFiles = AddTbbLibs(nonBuildFiles, pathBinArch)
% For Linux and Mac

arch = computer('arch');
if strcmpi(arch,'glnxa64')
   nonBuildFiles{end+1} = strcat(pathBinArch,'libtbb.so.2');
else % MAC
   nonBuildFiles{end+1} = strcat(pathBinArch,'libtbb.dylib');
end

%==========================================================================
function nonBuildFiles = AddCUDALibs(nonBuildFiles, pathBinArch)
% CUDA: required by all OpenCV libs when OpenCV is built WITH_CUDA=ON.
cudaLibs = {'cudart', 'nppc', 'nppc', 'nppial', 'nppicc', 'nppicom', ...
            'nppidei', 'nppif', 'nppig', 'nppim', 'nppist', 'nppisu', ...
            'nppitc','npps','cufft'};

arch = computer('arch');
switch arch
    case 'win32'
        % CUDA not enabled on win32
        cudaLibs = [];
    case 'win64'
        cudaLibs = strcat(cudaLibs, '64_*.dll');
    case 'glnxa64'
        cudaLibs = strcat('lib', cudaLibs, '.so.9.*');
    case 'maci64'
        cudaLibs = strcat('lib', cudaLibs, '.*.*.dylib');

    otherwise
        assert(false,[ arch ' operating system not supported']);
end

if ~strcmpi(arch,'win32')
    cudaLibs = lookupInBinDir(pathBinArch, cudaLibs);
    cudaLibs = strcat(pathBinArch,cudaLibs);
    for i = 1:numel(cudaLibs)
        nonBuildFiles{end+1} = cudaLibs{i}; %#ok<AGROW>
    end
end

%==========================================================================
function out = lookupInBinDir(pathBinArch, libs)

numLibs = numel(libs);

out = cell(1,numLibs);

for i = 1:numLibs
    info = dir(fullfile(pathBinArch, libs{i}));
    out{i} = info(1).name;
end

%==========================================================================
function nonBuildFiles = AddGLNXRTlibs(nonBuildFiles)
% Linux specific runtime libraries
arch = computer('arch');
sysosPath = fullfile(matlabroot,'sys','os',arch,filesep);
nonBuildFiles{end+1} = strcat(sysosPath,'libstdc++.so.6');
nonBuildFiles{end+1} = strcat(sysosPath,'libgcc_s.so.1');

%==========================================================================
function nonBuildFilesNoExt = AddDefaultOpenCVLibraries( ...
                                       nonBuildFilesNoExt, ocv_ver_no_dots)

nonBuildFilesNoExt{end+1} = strcat('opencv_core', ocv_ver_no_dots);
nonBuildFilesNoExt{end+1} = strcat('opencv_imgproc', ocv_ver_no_dots);

%==========================================================================
function nonBuildFilesNoExt = AddFeaturesLib(nonBuildFilesNoExt, ...
                                                     ocv_ver_no_dots)

nonBuildFilesNoExt{end+1} = strcat('opencv_features2d', ocv_ver_no_dots);

%==========================================================================
function nonBuildFilesNoExt = AddFlannLib(nonBuildFilesNoExt, ...
                                                  ocv_ver_no_dots)

nonBuildFilesNoExt{end+1} = strcat('opencv_flann', ocv_ver_no_dots);
