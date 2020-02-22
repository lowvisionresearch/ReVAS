function [default, validate, before, after, keyword, axesHandles] = GetDefaults(module)
% [default, validate, before, after, keyword, axesHandles] = GetDefaults(module)
%
%   Returns default parameter values and validation functions for each
%   corepipeline module. Case-insensitive. 'module' is a char array
%   representing the name of the corepipeline function.
%
%   In addition, it returns before and after cell arrays, indicating which
%   modules can preceed or succeed a given module. 'none' means original,
%   unprocessed video. It also returns the keyword to be used in
%   that module's output file name. Finally, it also returns tag names for
%   axes handles to be sent to these modules in GUI mode.
%
% Mehmet N. Agaoglu 1/19/2020 wrote initial version
% MNA               2/19/2020 added keyword.

% make lower case for more robust matches
module = lower(module);

% see if there is an .m extension, if so remove it
[~,module,~] = fileparts(module);

switch module
      
    case 'findblinkframes'
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.badFrames = false;
        default.axesHandles = [];
        default.stitchCriteria = 1;
        default.numberOfBins = 256;
        default.meanDifferenceThreshold = 10; 
        
        % validation functions
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.badFrames = @(x) all(islogical(x));
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.stitchCriteria = @IsPositiveInteger;
        validate.numberOfBins = @(x) IsPositiveInteger(x) & (x<=256);
        validate.meanDifferenceThreshold = @IsPositiveRealNumber;  
        
        % list which modules can preceed or succeed this one
        before = {'trimvideo','removestimuli','gammacorrect'};
        after = {'none','trimvideo','removestimuli','gammacorrect','bandpassfilter'};
        
        % keyword to be used in filenames
        keyword = 'blinkframes';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'blinkAx'};
        
    case 'trimvideo'
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.badFrames = false;
        default.borderTrimAmount = [0 0 12 0];
        default.axesHandles = [];
        
        % validation functions
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.badFrames = @(x) all(islogical(x));
        validate.borderTrimAmount = @(x) all(IsNaturalNumber(x)) & (length(x)==4);
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        
        % list which modules can preceed or succeed this one
        before = {'none','trimvideo','removestimuli','gammacorrect','bandpassfilter','findblinkframes'};
        after = {'none','trimvideo','removestimuli','gammacorrect','bandpassfilter','findblinkframes','stripanalysis'};
        
        % keyword to be used in filenames
        keyword = 'trim';
        
        axesHandles = {'imAx'};
        
    case 'removestimuli'
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.badFrames = false;
        default.axesHandles = [];
        default.minPeakThreshold = 0.6;
        default.frameRate = 30;
        default.fillingMethod = 'resample';
        default.removalAreaSize = [];
        default.stimulus = [];
        default.stimulusSize = 11;
        default.stimulusThickness = 1;
        default.stimulusPolarity = 1;
        
        % validation functions 
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.badFrames = @(x) all(islogical(x));
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.minPeakThreshold = @IsNonNegativeRealNumber;
        validate.frameRate = @IsPositiveRealNumber;
        validate.fillingMethod = @(x) any(contains({'resample','noise'},x));
        validate.removalAreaSize = @(x) isempty(x) | (isnumeric(x) & all(IsPositiveRealNumber(x)) & length(x)==2);
        validate.stimulus = @(x) isempty(x) | ischar(x) | ((isnumeric(x) | islogical(x)) & size(x,1)>1 & size(x,2)>1 & size(x,3)==1);
        validate.stimulusSize = @IsPositiveInteger;
        validate.stimulusThickness = @IsPositiveInteger;
        validate.stimulusPolarity = @(x) islogical(x) | (isnumeric(x) & any(x == [0 1]));
        
        % list which modules can preceed or succeed this one
        before = {'none','trimvideo','findblinkframes'};
        after = {'none','trimvideo','findblinkframes','gammacorrect','bandpassfilter','stripanalysis'};
     
        % keyword to be used in filenames
        keyword = 'nostim';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'imAx','peakAx','stimAx'};
        
    case 'gammacorrect'
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.method = 'simpleGamma';
        default.gammaExponent = 0.6;
        default.toneCurve = uint8(0:255); 
        default.histLevels = 64;
        default.badFrames = false;
        default.axesHandles = [];
        
        % validation functions 
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.method = @(x) any(contains({'simpleGamma','histEq','toneMapping'},x));
        validate.gammaExponent = @IsNonNegativeRealNumber;
        validate.toneCurve = @(x) isa(x, 'uint8') & length(x)==256;
        validate.histLevels = @IsPositiveInteger;
        validate.badFrames = @(x) all(islogical(x));  
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        
        % list which modules can preceed or succeed this one
        before = {'none','trimvideo','findblinkframes','removestimuli','gammacorrect','bandpassfilter'};
        after = {'none','trimvideo','findblinkframes','gammacorrect','bandpassfilter','stripanalysis'};
        
        % keyword to be used in filenames
        keyword = 'gamscaled';
        
        axesHandles = {'imAx'};
        
    case 'bandpassfilter'
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.badFrames = false;
        default.smoothing = 1;
        default.lowSpatialFrequencyCutoff = 3;
        default.axesHandles = [];
        
        % validation functions
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.badFrames = @(x) all(islogical(x));
        validate.smoothing = @IsPositiveRealNumber;
        validate.lowSpatialFrequencyCutoff = @IsNonNegativeRealNumber;
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        
        % list which modules can preceed or succeed this one
        before = {'none','trimvideo','findblinkframes','removestimuli','gammacorrect','bandpassfilter'};
        after = {'none','trimvideo','gammacorrect','bandpassfilter','stripanalysis'};
        
        % keyword to be used in filenames
        keyword = 'bandfilt';
        
        axesHandles = {'imAx'};
        
    case 'pixel2degree'
        % default values
        default.overwrite = false;
        default.fov = 10;
        default.frameWidth = 512;
        
        % validation functions
        validate.overwrite = @islogical;
        validate.fov = @(x) isnumeric(x) & (x>0) & (x<100);
        validate.frameWidth = @IsPositiveInteger;
        
        % list which modules can preceed or succeed this one
        before = {'stripanalysis','rereference'};
        after = {'none','filtereyeposition','findsaccadesanddrifts','degree2pixel'};
        
        % keyword to be used in filenames
        keyword = 'deg';
        
        axesHandles = {};
        
    case 'degree2pixel'
        % default values
        default.overwrite = false;
        default.fov = 10;
        default.frameWidth = 512;
        
        % validation functions
        validate.overwrite = @islogical;
        validate.fov = @(x) isnumeric(x) & (x>0) & (x<100);
        validate.frameWidth = @IsPositiveInteger;
        
        % list which modules can preceed or succeed this one
        before = {'filtereyeposition','pixel2degree'};
        after = {'none','makereference','rereference'};
        
        % keyword to be used in filenames
        keyword = 'px';
        
        axesHandles = {};
    
    case 'stripanalysis' 
        % default values
        default.overwrite = false;
        default.enableGPU = false;
        default.enableVerbosity = 'none';
        default.dynamicReference = true;
        default.goodFrameCriterion = 0.8;
        default.swapFrameCriterion = 0.8;
        default.corrMethod = 'mex';
        default.referenceFrame = 1;
        default.badFrames = false;
        default.stripHeight = 11;
        default.stripWidth = [];
        default.samplingRate = 540;
        default.minPeakThreshold = 0.65;
        default.maxMotionThreshold = 0.12; % proportion of frame size
        default.adaptiveSearch = true;
        default.searchWindowHeight = 79;
        default.lookBackTime = 20;
        default.frameRate = 30;
        default.axesHandles = [];
        default.neighborhoodSize = 5;
        default.subpixelDepth = 0;
        default.trim = [0 0];

        % validation functions 
        validate.overwrite = @islogical;
        validate.enableGPU = @islogical;
        validate.enableVerbosity = @(x) CategoryOrLogicalOrNumeric(x,{'none','video','frame','strip'});
        validate.dynamicReference = @islogical;
        validate.goodFrameCriterion = @(x) IsPositiveRealNumber(x) & (x<=1);
        validate.swapFrameCriterion = @(x) IsPositiveRealNumber(x) & (x<=1);
        validate.corrMethod = @(x) any(contains({'mex','normxcorr','fft','cuda'},x));
        validate.referenceFrame = @(x) isscalar(x) | ischar(x) | (isnumeric(x) & size(x,1)>1 & size(x,2)>1);
        validate.badFrames = @(x) all(islogical(x));
        validate.stripHeight = @IsPositiveInteger;
        validate.stripWidth = @(x) isempty(x) | IsPositiveInteger(x);
        validate.samplingRate = @IsNaturalNumber;
        validate.minPeakThreshold = @IsNonNegativeRealNumber; 
        validate.maxMotionThreshold = @IsPositiveRealNumber;
        validate.adaptiveSearch = @islogical;
        validate.searchWindowHeight = @IsPositiveInteger;
        validate.lookBackTime = @(x) IsPositiveRealNumber(x) & (x>=2);
        validate.frameRate = @IsPositiveRealNumber;
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.neighborhoodSize = @IsPositiveInteger;
        validate.subpixelDepth = @IsNaturalNumber;
        validate.trim = @(x) all(IsNaturalNumber(x)) & (length(x)==2);
        
        % list which modules can preceed or succeed this one
        before = {'none','trimvideo','removestimuli','gammacorrect','bandpassfilter','findblinkframes','makereference'};
        after = {'none','stripanalysis','pixel2degree','makereference','rereference'};
        
        % keyword to be used in filenames
        keyword = 'position';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'imAx','peakAx','motAx','posAx'};
        
    case 'makereference'
        
        % default values
        default.overwrite = false;
        default.enableVerbosity = 'none';
        default.badFrames = false;
        default.rowNumbers = []; % fail, if not provided
        default.position = []; % fail, if not provided
        default.timeSec = []; % fail, if not provided
        default.peakValueArray = []; % fail, if not provided
        default.oldStripHeight = []; % fail, if not provided
        default.newStripHeight = 3;
        default.newStripWidth = [];
        default.axesHandles = [];
        default.subpixelForRef = 0;
        default.minPeakThreshold = 0.5;
        default.maxMotionThreshold = 0.06; % proportion of frame size
        default.trim = [0 0];
        default.enhanceStrips = true;
        default.makeStabilizedVideo = false;
        
        % validation functions 
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) CategoryOrLogicalOrNumeric(x,{'none','video','frame'});
        validate.badFrames = @(x) all(islogical(x));
        validate.rowNumbers = @(x) (length(x)>=1 & IsPositiveInteger(x));
        validate.position = @(x) (isnumeric(x) & size(x,1)>=1 & size(x,2)==2);
        validate.timeSec = @(x) (isnumeric(x) & size(x,1)>=1 & size(x,2)==1);
        validate.peakValueArray = @IsNonNegativeRealNumber;
        validate.oldStripHeight = @IsPositiveInteger;
        validate.newStripHeight = @IsPositiveInteger;
        validate.newStripWidth = @(x) isempty(x) | IsPositiveInteger(x);
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.subpixelForRef = @IsNaturalNumber;
        validate.minPeakThreshold = @IsNonNegativeRealNumber;
        validate.maxMotionThreshold = @(x) IsPositiveRealNumber(x) & (x<=1);
        validate.trim = @(x) all(IsNaturalNumber(x)) & (length(x)==2);
        validate.enhanceStrips = @islogical;
        validate.makeStabilizedVideo = @islogical;
        
        % list which modules can preceed or succeed this one
        before = {'degree2pixel','stripanalysis'};
        after = {'none','stripanalysis'};
        
        % keyword to be used in filenames
        keyword = 'reference';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'imAx','peakAx','motAx'};
        
    case 'rereference'
        
        % default values
        default.enableGPU = false;
        default.reRefCorrMethod = 'mex';
        default.enableVerbosity = false;
        default.fixTorsion = false;
        default.tilts = -5:.25:5;
        default.anchorRowNumber = [];
        default.anchorStripHeight = 15;
        default.anchorStripWidth = [];
        default.axesHandles = [];
        default.globalRefArgument = [];
        default.referenceFrame = [];
        
        % validation functions
        validate.enableGPU = @islogical;
        validate.reRefCorrMethod = @(x) any(contains({'mex','normxcorr'},x));
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.fixTorsion = @islogical;
        validate.tilts = @(x) IsRealNumber(x) & isvector(x);
        validate.anchorStripHeight = @IsPositiveInteger;
        validate.anchorRowNumber = @(x) isempty(x) | IsPositiveInteger(x);
        validate.anchorStripWidth = @(x) isempty(x) | IsPositiveInteger(x);
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.globalRefArgument = @(x) (ischar(x) | (isnumeric(x) & size(x,1)>1 & size(x,2)>1)) & ~islogical(x);
        validate.referenceFrame = @(x) (ischar(x) | (isnumeric(x) & size(x,1)>1 & size(x,2)>1)) & ~islogical(x);
        
        % list which modules can preceed or succeed this one
        before = {'degree2pixel','stripanalysis'};
        after = {'none','pixel2degree'};
        
        % keyword to be used in filenames
        keyword = 'reref';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'imAx'};
        
    case 'filtereyeposition'
        
        % default values
        default.overwrite = false;
        default.enableVerbosity = false;
        default.maxGapDurationMs = 20; % msec
        default.maxPosition = 10; % deg
        default.maxVelocity = 500; % deg/sec
        default.beforeAfterMs = 1; % msec
        default.medfilt1 = 7;
        default.sgolayfilt = [3 21];
        default.notch1 = [29 31 2];
        default.notch2 = [59 61 2];
        default.samplingRate = [];
        default.axesHandles = [];
   
        % validation functions
        validate.overwrite = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.maxGapDurationMs = @IsNonNegativeRealNumber;
        validate.maxPosition = @IsNonNegativeRealNumber;
        validate.maxVelocity = @IsNonNegativeRealNumber;
        validate.beforeAfterMs = @IsNonNegativeRealNumber;
        validate.medfilt1 = @(x) (IsNonNegativeRealNumber(x) & length(x)==1);
        validate.sgolayfilt = @(x) isempty(x) | (IsNonNegativeRealNumber(x) & length(x)==2);
        validate.notch1 = @(x) isempty(x) | (IsNonNegativeRealNumber(x) & length(x)==3);
        validate.notch2 = @(x) isempty(x) | (IsNonNegativeRealNumber(x) & length(x)==3);
        validate.samplingRate = @(x) isempty(x) | IsPositiveInteger(x);
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        
        % list which modules can preceed or succeed this one
        before = {'pixel2degree'};
        after = {'none','degree2pixel','findsaccadesanddrifts'};

        % keyword to be used in filenames
        keyword = 'filtered';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'posAx'};
        
    case 'findsaccadesanddrifts'
        
        % default values
        default.enableVerbosity = false;
        default.algorithm = 'hybrid';
        default.velocityMethod = 2;
        default.axesHandles = [];
        default.minInterSaccadeInterval = 20; % ms
        default.minSaccadeAmplitude = 0.06; % deg
        default.maxSaccadeAmplitude = 10; % deg
        default.minSaccadeDuration = 8; % ms
        default.maxSaccadeDuration = 100; % ms
        default.velocityThreshold = 30; % deg/sec
        default.lambdaForPeak = 8;
        default.windowSize = 200; % ms
        default.lambdaForOnsetOffset = 3;
        
        % validation functions
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x >= 0);
        validate.algorithm = @(x) all(contains(x,{'hybrid','ivt','ek'}));
        validate.velocityMethod = @(x) isa(x,'function_handle') | (isscalar(x) & any([1 2],x));
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.minInterSaccadeInterval = @IsNonNegativeRealNumber;
        validate.minSaccadeAmplitude = @IsNonNegativeRealNumber;
        validate.maxSaccadeAmplitude = @IsNonNegativeRealNumber;
        validate.minSaccadeDuration = @IsNonNegativeRealNumber;
        validate.maxSaccadeDuration = @IsNonNegativeRealNumber;
        validate.velocityThreshold = @IsNonNegativeRealNumber;
        validate.lambdaForPeak = @IsNonNegativeRealNumber;
        validate.windowSize = @IsNonNegativeRealNumber;
        validate.lambdaForOnsetOffset = @IsNonNegativeRealNumber;
        
        % list which modules can preceed or succeed this one
        before = {'pixel2degree','filtereyeposition'};
        after = {'none','findsaccadesanddrifts'};
        
        % keyword to be used in filenames
        keyword = 'sacsanddrifts';
        
        % axes handle tags.. useful only in GUI mode
        axesHandles = {'posAx'};
        
    case 'none'
        
        default = [];
        validate = [];
        before = {'trimvideo','removestimuli','findblinkframes',...
            'bandpassfilter','stripanalysis','pixel2degree','degree2pixel',...
            'rereference','makereference','filtereyeposition','findsaccadesanddrifts'};
        after = before;
        axesHandles = [];
        
    otherwise
        error('GetDefaults: unknown module name.');
end




function tf = CategoryOrLogicalOrNumeric(x,categories)

if ischar(x)
    tf = any(contains(categories,x));
    return;
end

tf = islogical(x) | isnumeric(x);
    
