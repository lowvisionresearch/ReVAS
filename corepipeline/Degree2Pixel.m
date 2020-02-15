function [outputArgument, params] = Degree2Pixel(inputArgument, params)
% [outputArgument, params] = Degree2Pixel(inputArgument, params)
%   
%   A routine to convert position signals from degree units to pixels. Also
%   represents a simple example of how to develop additional modules for
%   a custom pipeline. 
% 
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputArgument| is a file path for the eye position data that has to
%   have two arrays, |positionDeg| and |timeSec|. Or, if it is not 
%   a file path, then it must be a nxm double array, where m>=2 and n is
%   the number of data points. The last column of |inputArgument| is always
%   treated as the time signal. The other columns are treated as eye
%   position signals, which will be subjected to scaling.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   fov                 : field of view in degrees.
%   frameWidth          : width of the original video frame in pixels.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputArgument| is the same type as inputArgument but after scaling.
%   If inputArgument is an array of eye positions and time, then
%   outputArgument is also an array of eye positions in pixels and time. If
%   inputArgument is a file containin eye positions, then outputArgument is
%   the same file containing eye positions in pixels (position). 
%
%   |params| structure.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       params = struct;
%       inputArray = [eyePosition time];
%       outputArray = Degree2Pixel(inputArray, params);
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputPath = 'MyFile.mat';
%       params.fov = 10;
%       params.frameWidth = 512;
%       outputPath = Degree2Pixel(inputPath, params);
% 
% MNA 2/15/2020


%% Determine inputType type.
if ischar(inputArgument)
    % A path was passed in.
    % Read and once finished with this module, write the result.
    writeResult = true;
else
    % A video matrix was passed in.
    % Do not write the result; return it instead.
    writeResult = false;
end


%% Set parameters to defaults if not specified.

if nargin < 2 
    params = struct;
end

% validate params
[~,callerStr] = fileparts(mfilename);
[default, validate] = GetDefaults(callerStr);
params = ValidateField(params,default,validate,callerStr);


%% Handle overwrite scenarios.
if writeResult
    outputFilePath = inputArgument;
    params.outputFilePath = outputFilePath;
end

%% Handle inputArgument scenarios
if writeResult
    % check if input file exists
    if exist(inputArgument,'file') 
        [~,~,ext] = fileparts(inputArgument);
        if strcmp(ext,'.mat')
            % load the data
            load(inputArgument,'positionDeg','timeSec');
        end
    end
    
    if isfield(params,'positionDeg')
        positionDeg = params.positionDeg;
        timeSec = params.timeSec;
    end
    
    if ~exist('positionDeg','var')
        error('Degree2Pixel: eye position array cannot be found!');   
    end
    
    position = positionDeg / (params.fov/params.frameWidth);

else
    % inputArgument is not a file path, but carries the eye position data.
    % last column is always 'time'
    position = inputArgument(:,1:size(inputArgument,2)-1) / (params.fov/params.frameWidth);
    timeSec = inputArgument(:,size(inputArgument,2));
end


%% Save converted data.
if writeResult 
    
    % remove unnecessary fields
    params = RemoveFields(params,{'logBox','axesHandles'}); 
    
    % append the file with converted position traces
    save(outputFilePath,'position','-append');
    
    outputArgument = outputFilePath;
else
    outputArgument = [position timeSec];
end


