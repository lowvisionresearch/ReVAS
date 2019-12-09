function [default, validate] = GetDefaults(module)

switch module
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
        validate.badFrames = @(x) all(logical(x));
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
        
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
%         
%     case
        
    otherwise
        error('GetDefaults: unknown module name.');
end



