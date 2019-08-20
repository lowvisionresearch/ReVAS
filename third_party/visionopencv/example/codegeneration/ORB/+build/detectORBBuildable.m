classdef detectORBBuildable < coder.ExternalDependency %#codegen
    % This class encapsulates the interface between external code and
    % MATLAB code intended for code generation. This class contains
    % information about external file locations, build information, and the
    % programming interface to external functions.

    % Copyright 2015-2017 The MathWorks, Inc.    

    methods (Static)

        %==================================================================
        function name = getDescriptiveName(~)
            name = 'detectORBBuildable';
        end

        %==================================================================
        function b = isSupportedContext(context)
            % Supports code generation on MATLAB host target
            b = context.isMatlabHostTarget();
        end

        %==================================================================
        function updateBuildInfo(buildInfo, context)

            mainDir = fullfile(fileparts(which('mexOpenCV.m')), ...
                               'example','codegeneration','ORB');

            % Headers
            buildInfo.addIncludePaths({fullfile(matlabroot,'toolbox', ...
                'vision','builtins','src','ocv','include'), ...
                fullfile(matlabroot,'toolbox', ...
                'vision','builtins','src','ocvcg','opencv','include'), ...
                fullfile(mainDir, 'include')} );
            buildInfo.addIncludeFiles({'vision_defines.h', ...
                                       'cgCommon.hpp', ...
                                       'utilityORB.hpp', ...
                                       'detectORBCore_api.hpp'});
            % Sources
            buildInfo.addSourcePaths({fullfile(mainDir, 'source')});
            buildInfo.addSourceFiles({'detectORBFeaturesOCVcodegen.cpp', ...
                                      'utilityORB.cpp'});

            % Link objects
            build.ORBOpenCVBuildInfo(buildInfo, context);
        end

        %==================================================================
        % Method to call the external code
        function points = detectORBFeatures(Iu8)
            % The number of features detected by ORB is determined at
            % run-time. OpenCV creates memory to hold the output features.
            %
            % The code below handles transformation of the data types from
            % OpenCV to the ones that can be interpreted by the MATLAB
            % Coder. Additionally, it handles passing the output size
            % information from OpenCV to MATLAB Coder.

            coder.inline('always');
            % Include header file in generated code.
            coder.cinclude('detectORBCore_api.hpp');

            % Declare output variable with type supported by OpenCV.
            ptrKeypoints = coder.opaque('void *', 'NULL');

            % Setup parameters to call external function
            numOut = int32(0);

            nRows = int32(size(Iu8,1));
            nCols = int32(size(Iu8,2));

            isRGB = ~ismatrix(Iu8);

            % Detect ORB Features and determine the output sizes using
            % external function call.
            numOut(1)=coder.ceval('detectORB_detect',...
              coder.ref(Iu8), ...
              nRows, nCols, isRGB, ...
              coder.ref(ptrKeypoints));

            % Setup output parameters with unbounded size.
            % Since output size is determined at run-time by OpenCV,
            % declare the outputs to be unbounded.
            coder.varsize('location',[inf 2]);
            coder.varsize('metric',[inf 1]);
            coder.varsize('scale',[inf 1]);
            coder.varsize('orientation',[inf 1]);
            coder.varsize('misc',[inf 1]);

            % Preallocate output memory dynamically with size determined at
            % run-time.
            location    = coder.nullcopy(zeros(numOut,2,'single'));
            metric      = coder.nullcopy(zeros(numOut,1,'single'));
            scale       = coder.nullcopy(zeros(numOut,1,'single'));
            orientation = coder.nullcopy(zeros(numOut,1,'single'));
            misc        = coder.nullcopy(zeros(numOut,1,'int32'));

            % Copy detected ORB Features from memory held by OpenCV to
            % buffers preallocated by MATLAB Coder.
            coder.ceval('detectORB_assignOutput', ptrKeypoints, ...
                coder.ref(location),...
                coder.ref(metric),...
                coder.ref(scale),...
                coder.ref(orientation),...
                coder.ref(misc));

            points.Location = location;
            points.Metric   = metric;
            points.Scale    = scale;
            points.Orientation = orientation;
            points.Misc    = misc;
        end
    end
end
