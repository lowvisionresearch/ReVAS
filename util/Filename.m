function outputFilePath = Filename(inputFilePath, moduleToApply, varargin)
% Filename
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  varargin{1} can be samplingRate for 'strip' and iteration for 'ref'.
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
    optional = varargin{1};
end

% Deconstruct input file path.
[inputDir, inputFileName, inputExtension] = fileparts(inputFilePath);

% Assume input had an  avi extension if none provided.
if isempty(inputExtension)
    inputExtension = '.avi';
end

switch moduleToApply
    case 'trim'
        outputFilePath = fullfile(inputDir, [inputFileName '_trim' inputExtension]);
    case 'removestim'
        outputFilePath = fullfile(inputDir, [inputFileName '_nostim' inputExtension]);
    case 'blink'
        outputFilePath = fullfile(inputDir, [inputFileName '_blinkframes.mat']);
    case 'stimlocs'
        outputFilePath = fullfile(inputDir, [inputFileName '_stimlocs.mat']);
    case 'gamma'
        outputFilePath = fullfile(inputDir, [inputFileName '_gamscaled' inputExtension]);
    case 'bandpass'
        outputFilePath = fullfile(inputDir, [inputFileName '_bandfilt' inputExtension]);
    case 'ref'
        outputFilePath = fullfile(inputDir, [inputFileName '_reference.mat']);
    case 'strip'
        % Set samplingRate to default value if necessary
        if nargin < 3
            optional = 540;
        end
        outputFilePath = fullfile(inputDir, [inputFileName '_' num2str(optional) '_hz_position.mat']);
    case 'filtered'
        outputFilePath = fullfile(inputDir, [inputFileName '_filtered.mat']);
    case 'reref'
        outputFilePath = fullfile(inputDir, [inputFileName '_reref' inputExtension]);
    case 'sacsdrifts'
        outputFilePath = fullfile(inputDir, [inputFileName '_sacsdrifts.mat']);
end

