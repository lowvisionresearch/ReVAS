function [default, validate] = GetDefaults(module)

switch module
      
    case 'FindBlinkFrames'
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
        
    case 'TrimVideo'
        % default values
        default.overwrite = false;
        default.badFrames = false;
        default.borderTrimAmount = [0 0 12 0];
        
        % validation functions
        validate.overwrite = @islogical;
        validate.badFrames = @(x) all(islogical(x));
        validate.borderTrimAmount = @(x) all(IsNaturalNumber(x)) & (length(x)==4);
        
    case 'RemoveStimuli'
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
     
    case 'GammaCorrect'
        % default values
        default.overwrite = false;
        default.method = 'simpleGamma';
        default.gammaExponent = 2.2;
        default.toneCurve = uint8(0:255); 
        default.histLevels = 64;
        default.badFrames = false;
        
        % validation functions 
        validate.overwrite = @islogical;
        validate.method = @(x) any(contains({'simpleGamma','histEq','toneMapping'},x));
        validate.gammaExponent = @IsNonNegativeRealNumber;
        validate.toneCurve = @(x) isa(x, 'uint8') & length(x)==256;
        validate.histLevels = @IsPositiveInteger;
        validate.badFrames = @(x) all(islogical(x));  
        
    case 'BandpassFilter'
        % default values
        default.overwrite = false;
        default.badFrames = false;
        default.smoothing = 1;
        default.lowSpatialFrequencyCutoff = 3;
        
        % validation functions
        validate.overwrite = @islogical;
        validate.badFrames = @(x) all(islogical(x));
        validate.smoothing = @IsPositiveRealNumber;
        validate.lowSpatialFrequencyCutoff = @IsNonNegativeRealNumber;
    
    case 'StripAnalysis' 
        % default values
        default.overwrite = false;
        default.enableGPU = false;
        default.enableVerbosity = false;
        default.enableSubpixel = false;
        default.corrMethod = 'mex';
        default.referenceFrame = 1;
        default.badFrames = false;
        default.stripHeight = 11;
        default.samplingRate = 540;
        default.minPeakThreshold = 0.3;
        default.adaptiveSearch = true;
        default.searchWindowHeight = 79;
        default.lookBackTime = 10;
        default.frameRate = 30;
        default.axesHandles = [];
        default.neighborhoodSize = 5;
        default.subpixelDepth = 2;
        default.trim = [0 0];

        % validation functions 
        validate.overwrite = @islogical;
        validate.enableGPU = @islogical;
        validate.enableVerbosity = @(x) islogical(x) | (isscalar(x) & x>=0);
        validate.enableSubpixel = @islogical;
        validate.corrMethod = @(x) any(contains({'mex','normxcorr','fft','cuda'},x));
        validate.referenceFrame = @(x) isscalar(x) | ischar(x) | (isnumeric(x) & size(x,1)>1 & size(x,2)>1);
        validate.badFrames = @(x) all(islogical(x));
        validate.stripHeight = @IsNaturalNumber;
        validate.samplingRate = @IsNaturalNumber;
        validate.minPeakThreshold = @IsNonNegativeRealNumber;
        validate.adaptiveSearch = @islogical;
        validate.searchWindowHeight = @IsPositiveInteger;
        validate.lookBackTime = @(x) IsPositiveRealNumber(x) & (x>=2);
        validate.frameRate = @IsPositiveRealNumber;
        validate.axesHandles = @(x) isempty(x) | all(ishandle(x));
        validate.neighborhoodSize = @IsPositiveInteger;
        validate.subpixelDepth = @IsPositiveInteger;
        validate.trim = @(x) all(IsNaturalNumber(x)) & (length(x)==2);
        
    otherwise
        error('GetDefaults: unknown module name.');
end



