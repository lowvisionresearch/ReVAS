function [outputArgument, params] = Pixel2Degree(inputArgument, params)
% [outputArgument, params] = Pixel2Degree(inputArgument, params)
%   
%   A routine to convert position signals from pixel units to degrees. Also
%   represents a simple example of how to develop additional modules for
%   a custom pipeline. 
% 
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputArgument| is a file path for the eye position data that has to
%   have two arrays, |position| and |timeSec|. Or, if it is not 
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
%   overwrite           : true/false
%   fov                 : field of view in degrees.
%   frameWidth          : width of the original video frame in pixels.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputArgument| is the same type as inputArgument but after scaling.
%   If inputArgument is an array of eye positions and time, then
%   outputArgument is also an array of eye positions in degrees and time. If
%   inputArgument is a file containin eye positions, then outputArgument is
%   also a file containing eye positions in degrees (positionDeg). 
%
%   |params| structure.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       params = struct;
%       inputArray = [eyePosition time];
%       outputArray = Pixel2Degree(inputArray, params);
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputPath = 'MyFile.mat';
%       params.fov = 10;
%       params.frameWidth = 512;
%       outputPath = Pixel2Degree(inputPath, params);
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


%% Handle GUI mode
% params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
else
    logBox = [];
end

% params will have access to a uicontrol object in GUI mode. so if it does
% not already have that, create the field and set it to false so that this
% module can be used without the GUI
if ~isfield(params,'abort')
    params.abort.Value = false;
end


%% Handle overwrite scenarios.
if writeResult
    outputFilePath = inputArgument;
    params.outputFilePath = outputFilePath;
end

%% Handle overwrite scenarios
if writeResult
    outputFilePath = Filename(inputArgument, 'deg');
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['Pixel2Degree() did not execute because it would overwrite existing file. (' outputFilePath ')'], logBox);
        RevasMessage('Pixel2Degree() is returning results from existing file.',logBox); 
        outputArgument = outputFilePath;
        return;
    else
        RevasMessage(['Pixel2Degree() is proceeding and overwriting an existing file. (' outputFilePath ')'], logBox);  
    end
end
    
%% Handle inputArgument scenarios
if writeResult
    
    % check if input file exists
    if exist(inputArgument,'file') 
        [~,~,ext] = fileparts(inputArgument);
        if strcmp(ext,'.mat')
            % load the data
            load(inputArgument,'position','timeSec');
        end
    end
    
    if isfield(params,'position')
        position = params.position;
        timeSec = params.timeSec;
    end
    
    if ~exist('position','var')
        error('Pixel2Degree: eye position array cannot be found!');   
    end
    
    positionDeg = position * params.fov/params.frameWidth;

else
    % inputArgument is not a file path, but carries the eye position data.
    % last column is always 'time'
    positionDeg = inputArgument(:,1:size(inputArgument,2)-1) * params.fov/params.frameWidth;
    timeSec = inputArgument(:,size(inputArgument,2));
end


%% Save converted data.

% remove unnecessary fields
abort = params.abort.Value;
params = RemoveFields(params,{'logBox','axesHandles','abort'}); 

if writeResult && ~abort
    
    % append the file with converted position traces
    save(outputFilePath,'positionDeg','timeSec','params');
    
    outputArgument = outputFilePath;
else
    outputArgument = [positionDeg timeSec];
end


