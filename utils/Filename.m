function [outputFilePath, keyword, module] = Filename(inputFilePath, moduleToApply, varargin)
% [outputFilePath, keyword, module] = Filename(inputFilePath, moduleToApply, varargin)
%
%  Utility function for converting one file name to another, according to
%  the conventions used by ReVAS.
%
%  varargin{1} is params.
%
%  Returns output file path, keyword for the queried module, and name of
%  the module. 
%
%
%  MTS 8/23/2019 wrote the initial version
%  MNA 12/26/2019 modified according to new ReVAS guidelines.
%  MNA 2/19/2020 modified to return keyword and module names as well.
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
        keyword = 'trim';
        module = 'trimvideo';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword inputExtension]);
    case {'removestim','removestimuli','nostim'}
        keyword = 'nostim';
        module = 'removestimuli';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword inputExtension]);
    case {'blink','findblinkframes','blinkframes'}
        keyword = 'blinkframes';
        module = 'findblinkframes';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case 'stimlocs'
        keyword = 'stimlocs';
        module = 'removestimuli';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case {'gamma','gammacorrect','gamscaled'}
        keyword = 'gamscaled';
        module = 'gammacorrect';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword inputExtension]);
    case {'bandpass','bandpassfilter','bandfilt'}
        keyword = 'bandfilt';
        module = 'bandpassfilter';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword inputExtension]);
    case {'ref','makereference','reference'}
        keyword = 'reference';
        module = 'makereference';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case {'deg','pixel2degree'}
        keyword = 'deg';
        module = 'pixel2degree';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case {'px','degree2pixel'}
        keyword = 'px';
        module = 'degree2pixel';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case {'strip','stripanalysis','position'}
        keyword = 'position';
        module = 'stripanalysis';
        % Set samplingRate to default value if necessary
        if nargin < 3
            params.samplingRate = 540;
        end
        outputFilePath = fullfile(inputDir, [inputFileName '_' num2str(params.samplingRate) '_hz_' keyword '.mat']);
    case {'filtered','filtereyeposition'}
        keyword = 'filtered';
        module = 'filtereyeposition';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case {'reref','rereference'}
        keyword = 'reref';
        module = 'rereference';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword inputExtension]);
    case {'sacsdrifts','findsaccadesanddrifts'}
        keyword = 'sacsdrifts';
        module = 'findsaccadesanddrifts';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    case 'inmemory'
        keyword = 'inmemory';
        module = 'inmemory';
        outputFilePath = fullfile(inputDir, [inputFileName '_' keyword '.mat']);
    otherwise
        outputFilePath = inputFilePath;
        keyword = [];
        module = [];
end

