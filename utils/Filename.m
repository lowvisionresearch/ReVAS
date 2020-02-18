function outputFilePath = Filename(inputFilePath, moduleToApply, varargin)
% Filename
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  varargin{1} is params.
%
%  Options for moduleToApply are:
%   - trim
%   - removestim
%   - blink
%   - stimlocs
%   - gamma
%   - bandpass
%   - ref
%   - strip
%   - filtered
%   - reref
%   - sacsdrifts
%
%  MTS 8/23/19 wrote the initial version
%  MNA 12/26/19 modified according to new ReVAS guidelines.
%
%

if nargin > 2
    params = varargin{1};
end

% Deconstruct input file path.
[inputDir, inputFileName, inputExtension] = fileparts(inputFilePath);

% Assume input had an  avi extension if none provided.
if isempty(inputExtension)
    inputExtension = '.avi';
end

switch lower(moduleToApply)
    case {'trim','trimvideo'}
        outputFilePath = fullfile(inputDir, [inputFileName '_trim' inputExtension]);
    case {'removestim','removestimuli'}
        outputFilePath = fullfile(inputDir, [inputFileName '_nostim' inputExtension]);
    case {'blink','findblinkframes'}
        outputFilePath = fullfile(inputDir, [inputFileName '_blinkframes.mat']);
    case 'stimlocs'
        outputFilePath = fullfile(inputDir, [inputFileName '_stimlocs.mat']);
    case {'gamma','gammacorrect'}
        outputFilePath = fullfile(inputDir, [inputFileName '_gamscaled' inputExtension]);
    case {'bandpass','bandpassfilter'}
        outputFilePath = fullfile(inputDir, [inputFileName '_bandfilt' inputExtension]);
    case {'ref','makereference'}
        outputFilePath = fullfile(inputDir, [inputFileName '_reference.mat']);
    case {'strip','stripanalysis'}
        % Set samplingRate to default value if necessary
        if nargin < 3
            params.samplingRate = 540;
        end
        outputFilePath = fullfile(inputDir, [inputFileName '_' num2str(params.samplingRate) '_hz_position.mat']);
    case {'filtered','filtereyeposition'}
        outputFilePath = fullfile(inputDir, [inputFileName '_filtered.mat']);
    case {'reref','rereference'}
        outputFilePath = fullfile(inputDir, [inputFileName '_reref' inputExtension]);
    case {'sacsdrifts','findsaccadesanddrifts'}
        outputFilePath = fullfile(inputDir, [inputFileName '_sacsdrifts.mat']);
    case 'inmemory'
        outputFilePath = fullfile(inputDir, [inputFileName '_inmemory.mat']);
    otherwise
        outputFilePath = inputFilePath;
end

