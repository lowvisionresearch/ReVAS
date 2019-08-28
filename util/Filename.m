function outputFilePath = Filename(inputFilePath, moduleToApply, samplingRate)
% Filename
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  samplingRate is only required for useful traces (default 540 Hz).
%
%  Options for moduleToApply are:
%   - trim
%   - removestim
%   - blink
%   - stimlocs
%   - gamma
%   - bandpass
%   - coarseref
%   - framepos
%   - fineref
%   - usefultraces
%   - filtered
%   - reref
%   - sacsdrift
%
%  MTS 8/23/19 wrote the initial version

% Deconstruct input file path.
[inputDir, inputFileName, inputExtension] = fileparts(inputFilePath);

% Assume input had an  avi extension if none provided.
if isempty(inputExtension)
    inputExtension = '.avi';
end

switch moduleToApply
    case 'trim'
        outputFilePath = fullfile(inputDir, [inputFileName '_dwt' inputExtension]);
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
    case 'coarseref'
        outputFilePath = fullfile(inputDir, [inputFileName '_coarseref.mat']);
    case 'framepos'
        outputFilePath = 'framePositions.mat';
    case 'fineref'
        outputFilePath = fullfile(inputDir, [inputFileName '_refframe.mat']);
    case 'usefultraces'
        % Set samplingRate to default value if necessary
        if nargin < 3
            samplingRate = 540;
            RevasMessage('using default parameter for samplingRate');
        end
        outputFilePath = fullfile(inputDir, [inputFileName '_' num2str(samplingRate) '_hz_final.mat']);
    case 'filtered'
        outputFilePath = fullfile(inputDir, [inputFileName '_filtered.mat']);
    case 'reref'
        outputFilePath = fullfile(inputDir, [inputFileName '_reref' inputExtension]);
    case 'sacsdrifts'
        outputFilePath = fullfile(inputDir, [inputFileName '_sacsdrifts.mat']);
end

end