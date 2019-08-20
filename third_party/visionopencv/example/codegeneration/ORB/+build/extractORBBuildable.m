classdef extractORBBuildable < coder.ExternalDependency %#codegen
    % This class encapsulates the interface between external code and
    % MATLAB code intended for code generation. This class contains
    % information about external file locations, build information, and the
    % programming interface to external functions.

    % Copyright 2015-2017 The MathWorks, Inc.  

    methods (Static)

        %==================================================================
        function name = getDescriptiveName(~)
            name = 'extractORBFeatures';
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
                fullfile(mainDir, 'include'), ...
                fullfile(matlabroot,'toolbox', ...
                'vision','builtins','src','ocvcg', 'opencv', 'include')} );

            buildInfo.addIncludeFiles({'vision_defines.h', ...
                                       'extractORBCore_api.hpp', ...
                                       'utilityORB.hpp', ...
                                       'cgCommon.hpp'});

            % Sources
            buildInfo.addSourcePaths({fullfile(mainDir, 'source')});
            buildInfo.addSourceFiles({'extractORBFeaturesOCVcodegen.cpp', ...
                                      'utilityORB.cpp'});

            % Link objects
            build.ORBOpenCVBuildInfo(buildInfo, context);
        end

        %==================================================================
        % Method to call the external code
        function [features, valid_points] = extractORBFeatures(Iu8, points)
            % The number of features extracted by ORB is determined at
            % run-time. OpenCV creates memory to hold the output features.
            %
            % The code below handles transformation of the data types from
            % OpenCV to the ones that can be interpreted by the MATLAB
            % Coder. Additionally, it handles passing the output size
            % information from OpenCV to MATLAB Coder.

            coder.inline('always');
            % Include header file in generated code
            coder.cinclude('extractORBCore_api.hpp');

            % Declare output variable with type supported by OpenCV.
            ptrKeypoints = coder.opaque('void *', 'NULL');
            ptrFeatures  = coder.opaque('void *', 'NULL');

            % Setup parameters to call external function
            numOut = int32(0);

            nRows = int32(size(Iu8,1));
            nCols = int32(size(Iu8,2));

            inLocation    = points.Location;
            inMetric      = points.Metric;
            inScale       = points.Scale;
            inOrientation = points.Orientation;
            inMisc        = points.Misc;

            % Extract ORB Features and determine the output sizes using
            % external function call
            numOut(1) = coder.ceval('extractORB_compute',...
                coder.ref(Iu8), ...
                nRows, nCols,...
                coder.ref(inLocation),...
                coder.ref(inMetric),...
                coder.ref(inScale),...
                coder.ref(inOrientation),...
                coder.ref(inMisc),...
                int32(size(inLocation,1)),...
                coder.ref(ptrFeatures), coder.ref(ptrKeypoints));

            % Setup output parameters with unbounded size.
            % Since output size is determined at run-time by OpenCV,
            % declare the outputs to be unbounded.
            coder.varsize('location',[inf 2]);
            coder.varsize('metric',[inf 1]);
            coder.varsize('scale',[inf 1]);
            coder.varsize('orientation',[inf 1]);
            coder.varsize('misc',[inf 1]);
            coder.varsize('features',[inf 64]);

            % Preallocate output memory dynamically with size determined at
            % run-time.
            location    = coder.nullcopy(zeros(numOut,2,'single'));
            metric      = coder.nullcopy(zeros(numOut,1,'single'));
            scale       = coder.nullcopy(zeros(numOut,1,'single'));
            misc        = coder.nullcopy(zeros(numOut,1,'int32'));
            orientation = coder.nullcopy(zeros(numOut,1,'single'));
            features    = coder.nullcopy(zeros(numOut,64,'uint8'));

            % Copy extracted ORB Features from memory held by OpenCV to
            % buffers preallocated by MATLAB Coder.
            coder.ceval('extractORB_assignOutput', ...
                ptrFeatures, ptrKeypoints, ...
                coder.ref(location),...
                coder.ref(metric),...
                coder.ref(scale),...
                coder.ref(orientation),...
                coder.ref(misc),...
                coder.ref(features));

            valid_points.Location = location;
            valid_points.Metric   = metric;
            valid_points.Scale    = scale;
            valid_points.Misc     = misc;
            valid_points.Orientation = orientation;
        end
    end
end
