function outputFileName = Filename(inputFileName, moduleToApply, samplingRate)
% Filename
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  samplingRate is only required for useful traces (default 540 Hz).
%
%  Options for moduleToApply are:
%   - trim
%   - removestim
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

% Check to see if there is a file extension on inputFileName.
if length(inputFileName) > 4 && inputFileName(end-3) == '.'
    extension = inputFileName(end-3:end);
    inputFileName = inputFileName(1:end-4);
else
    extension = '.avi';
end

switch moduleToApply
    case 'trim'
        outputFileName = [inputFileName '_dwt' extension];
    case 'removestim'
        outputFileName = [inputFileName '_nostim' extension];
    case 'stimlocs'
        outputFileName = [inputFileName '_stimlocs.mat'];
    case 'gamma'
        outputFileName = [inputFileName '_gamscaled' extension];
    case 'bandpass'
        outputFileName = [inputFileName '_bandfilt' extension];
    case 'coarseref'
        outputFileName = [inputFileName '_coarseref.mat'];
    case 'framepos'
        outputFileName = 'framePositions.mat';
    case 'fineref'
        outputFileName = [inputFileName '_refframe.mat'];
    case 'usefultraces'
        % Set samplingRate to default value if necessary
        if nargin < 3
            samplingRate = 540;
            RevasMessage('using default parameter for samplingRate');
        end
        outputFileName = [inputFileName '_' num2str(samplingRate) '_hz_final.mat'];
    case 'filtered'
        outputFileName = [inputFileName '_filtered.mat'];
    case 'reref'
        outputFileName = [inputFileName '_reref.mat'];
    case 'sacsdrifts'
        outputFileName = [inputFileName '_sacsdrifts.mat'];
end

end