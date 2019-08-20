function mexOpenCV(varargin)
%mexOpenCV Build custom C++ MEX-function based on OpenCV
%
%  Usage:
%     mexOpenCV [options ...] file [files ...]
%
%  Description:
%     The mexOpenCV function is an extension of the MATLAB mex function.
%     It compiles and links source files into a MATLAB executable mex-file.
%     mexOpenCV automatically links to a subset of OpenCV libraries and
%     includes routines to convert between MATLAB and OpenCV data types.
%
%     Included OpenCV libraries:
%     opencv_calib3d, opencv_core, opencv_cudaarithm, opencv_cudabgsegm,
%     opencv_cudafeatures2d, opencv_cudafilters, opencv_cudaimgproc,
%     opencv_cudalegacy, opencv_cudaobjdetect, opencv_cudaoptflow,
%     opencv_cudastereo, opencv_cudawarping, opencv_cudev,
%     opencv_features2d, opencv_flann, opencv_imgproc, opencv_ml,
%     opencv_objdetect, opencv_photo, opencv_shape, opencv_stitching,
%     opencv_superres, opencv_video, opencv_videostab
%
%     Libraries that are not included:
%     opencv_highgui, opencv_imgcodecs, opencv_videoio, opencv_cudacodec
%
%     The conversion routines are defined in
%           (matlabroot)\extern\include\opencvmex.hpp.
%
%  Command Line Options Available on All Platforms:
%     See <a href="matlab:doc('mex')">mex function documentation</a> to learn about all available options.
%
%  Example - Create mex function calling OpenCV template matching routine
%  ----------------------------------------------------------------------
%  % Change folder to the location of matchTemplateOCV.cpp
%  baseFolder = fileparts(which('mexOpenCV.m'));
%  cd(fullfile(baseFolder,'example','TemplateMatching'));
%
%  % Create matchTemplateOCV.<mex file extension> in current folder with
%  % optional debug and verbose flags.
%  mexOpenCV matchTemplateOCV.cpp -g -v
%
%  % Test the generated mex file.
%  testMatchTemplate
%
%  See also mex

% Copyright 2014-2017 The MathWorks, Inc.

arch = computer('arch');
ocvcgDir = fullfile(matlabroot,'toolbox','vision','builtins','src','ocvcg');

% register message catalog
registerMessageCatalog();

%% Fill in structure describing items needed to build OpenCV based MEX file
cvstocvutil.include = fullfile(matlabroot, 'extern','include');
ocvconfig.include   = fullfile(ocvcgDir,'opencv','include');

%% Check compiler compatibility
% Get the name of compiler from mex command; Error for incompatible compiler
[isCompatCompiler, compilerForMexName, compilerUsedForOpenCVlib] = vision.internal.checkOCVSupportedCompiler(arch);

if ~isCompatCompiler
    error(message('opencvinterface:opencv:incompatibleCompiler', ...
    compilerUsedForOpenCVlib, compilerForMexName));
end

%% Setup include paths
if ispc
    % Get the path for opencv .lib files from main product
    ocvconfig.sharedLibraries = fullfile(ocvcgDir,'opencv',arch,'lib');
    % Get the path for libmwocvmex.lib from Computer Vision System Toolbox
    libDir = 'microsoft';
    linkLibPathMS = fullfile(matlabroot,'extern','lib',computer('arch'),libDir);
    cvstocvutil.sharedLibraries = linkLibPathMS;
else
    % Get the path for opencv .so/.dylib files from Computer Vision System
    % Toolbox
    ocvconfig.sharedLibraries = fullfile(matlabroot, 'bin', arch);
    % Get the path for libmwocvmex{.so, .dylib} from Computer Vision System
    % Toolbox
    cvstocvutil.sharedLibraries = ocvconfig.sharedLibraries;
end
ocvconfig.outputDirectory = pwd;

%% Define library info
% Use OpenCV version 3.4.0
if ispc
    ocvVer = '340';
    prefix = 'opencv_';
    lFlag = '-l';
elseif ismac
    ocvVer = '.3.4.0';
    prefix = 'opencv_';
    lFlag = '-l';
else  % linux
    ocvVer = '.so.3.4.0';
    prefix = 'libopencv_';
    lFlag = '-l:';% use ":" to link against the full names of the libraries
end

%% Specify libraries to link against

% Link against a limited set of libraries. See help text for unsupported
% libraries:
libs = {'calib3d', 'core', 'features2d', 'flann', 'imgproc', 'ml', ...
        'objdetect', 'photo', 'shape', 'stitching', 'superres', ...
        'video', 'videostab'};

dashLibs = {'mwocvmex'};

includes = {['-I' ocvconfig.include],['-I' cvstocvutil.include]};

% Add GPU related files and libraries

if exist(fullfile(matlabroot,'toolbox','distcomp'),'dir')
    libs = [libs {'cudaarithm', 'cudabgsegm', 'cudafeatures2d', ...
                  'cudafilters', 'cudaimgproc', 'cudalegacy', ...
                  'cudaobjdetect', 'cudaoptflow', 'cudastereo', ...
                  'cudawarping', 'cudev'}];
    includes = [includes ['-I' ...
        fullfile(toolboxdir('distcomp'),'gpu','extern','include')]];
end


libs          = strcat(prefix, libs, ocvVer);
versionedLibs = strcat(lFlag, libs);
dashLibs      = strcat('-l', dashLibs);


%% Build custom MEX function that links against OpenCV
try
    mex(includes{:},...
        ['-L' ocvconfig.sharedLibraries], ['-L' cvstocvutil.sharedLibraries],...
        dashLibs{:}, versionedLibs{:}, varargin{:} );
catch ME
    throw(ME);
end

%% ------------------------------------------------------------------------
function registerMessageCatalog()

thisDir = fileparts(mfilename('fullpath'));
resourceDir = [thisDir filesep '..' filesep '..' filesep '..'  filesep '..'];
[~] = registerrealtimecataloglocation(resourceDir);
