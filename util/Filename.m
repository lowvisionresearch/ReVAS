function outputFileName = Filename(inputFileName, moduleToApply, frequency)
% Filename
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  frequency is only required for useful traces.
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
   inputFileName = inputFileName(1:end-4);
end

switch moduleToApply
    case 'trim'
        outputFileName = [inputFileName '_dwt.avi'];
    case 'removestim'
        outputFileName = [inputFileName '_nostim.avi'];
    case 'stimlocs'
        outputFileName = [inputFileName '_stimlocs.mat'];
    case 'gamma'
        outputFileName = [inputFileName '_gamscaled.avi'];
    case 'bandpass'
        outputFileName = [inputFileName '_bandfilt.avi'];
    case 'coarseref'
        outputFileName = [inputFileName '_coarseref.mat'];
    case 'framepos'
        outputFileName = 'framePositions.mat';
    case 'fineref'
        outputFileName = [inputFileName '_refframe.mat'];
    case 'usefultraces'
        outputFileName = [inputFileName '_' frequency '_hz_final.mat'];
    case 'filtered'
        outputFileName = [inputFileName '_filtered.mat'];
    case 'reref'
        outputFileName = [inputFileName '_reref.mat'];
    case 'sacsdrift'
        outputFileName = [inputFileName '_sacsdrift.mat'];
end

end