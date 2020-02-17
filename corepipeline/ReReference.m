function [outputArgument, params, varargout] = ReReference(inputArgument, params)
% [outputArgument, params, varargout] = ReReference(inputArgument, params)
%REREFERENCE computes the offsets required to re-reference one reference
%   onto another.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputArgument| is a file path for the eye position data that has to
%   have two arrays, |position| and |timeSec|. Or, if it is not 
%   a file path, then it must be a nx3 double array, where n is
%   the number of data points. The last column of |inputArgument| is always
%   treated as the time signal. The other columns are treated as eye
%   position signals, which will be subjected to re-referencing. 
%   The result is that the re-referenced version of this file is stored with
%   '_reref' appended to the original file name. If |inputArgument| 
%   is not a file name but actual eye position data, then the re-referenced 
%   position data are not stored.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the 'params' structure
%   --------------------------------
%   overwrite               : set to true to overwrite existing files.
%                             Set to false to abort the function call if the
%                             files already exist. (default false)
%   referenceFrame          : local reference frame. is a 2D array 
%                             representing a local ref or a full path to a 
%                             file containing 'referenceFrame' or 
%                             'params.referenceFrame'.
%   globalRefArgument       : global reference frame. is a 2D array 
%                             representing a local ref or a full path to a 
%                             file containing 'referenceFrame' or 
%                             'params.referenceFrame'.
%   enableGPU               : a logical. if set to true, use GPU. (works for 
%                             'mex' method only.
%   enableVerbosity         : set to true to report back plots during
%                             execution. (default false)
%   corrMethod              : method to use for cross-correlation. you can 
%                             choose from 'normxcorr' for matlab's built-in
%                             normxcorr2, 'mex' for opencv's correlation, or 
%                             'fft' for our custom-implemented fast
%                             correlation method. 'cuda' is placed but not 
%                             implemented yet (default 'mex').
%   fixTorsion              : set to true if you want to improve
%                             cross-correlations by rotating the local ref
%                             with a range of tilts. (default true)
%   tilts                   : a 1D array with a set of tilt values.
%                             (relevant only when fixTorsion is true)
%                             (default -5:0.25:5)
%   anchorStripHeight       : height of the strip to be taken from localRef
%                             for rereferencing. (default 15)
%   anchorStripWidth        : width of the strip to be taken from localRef
%                             for rereferencing. if empty, uses localRef 
%                             width. default empty.
%   anchorRowNumber         : Row number in pixels from local reference.
%                             Strip(s) for rereferencing will be extracted
%                             starting from this index. If left empty,
%                             ReReference uses line contrasts to decide
%                             what's most informative strip. (default
%                             empty)
%   axesHandles             : axes handle for giving feedback. if not provided, 
%                             new figures are created. (relevant only when
%                             enableVerbosity is true)
%   
%   -----------------------------------
%   Output
%   -----------------------------------
%
%   |offset| is position shift (in pixels) required to reference the
%   position traces onto global ref.
%
%   |bestTilt| is the best tilt for localRef to be mapped onto globalRef
%
%   varargout{1} : offset
%   varargout{2} : bestTilt
%   varargout{3} : peakValues.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%   TO DO


%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% in GUI mode, params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
else
    logBox = [];
end


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

%% Handle GPU 

% check if CUDA enabled GPU exists
if params.enableGPU && (gpuDeviceCount < 1)
    params.enableGPU = false;
    RevasWarning('No supported GPU available. Rereference is reverting back to CPU', logBox);
end


%% Handle verbosity 

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Re-reference',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.67 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');
    params.axesHandles(1) = subplot(1,1,1);
    cla(params.axesHandles(1))
    tb = get(params.axesHandles(1),'toolbar');
    tb.Visible = 'on';
end


%% Handle overwrite scenarios.

if writeResult
    outputFilePath = Filename(inputArgument, 'reref');
    params.outputFilePath = outputFilePath;
    
    if ~exist(outputFilePath, 'file')
        % left blank to continue without issuing RevasMessage in this case
    elseif ~params.overwrite
        RevasMessage(['ReReference() did not execute because it would overwrite existing file. (' outputFilePath ')'], logBox);
        RevasMessage('ReReference() is returning results from existing file.',logBox); 
        
        % try loading existing file contents
        load(outputFilePath,'position','timeSec','offset','bestTilt','params','peakValues');
        outputArgument = outputFilePath;
        params.position = position;
        params.timeSec = timeSec;
        params.offset = offset;
        params.bestTilt = bestTilt;
        if nargout > 2
            varargout{1} = offset;
        end
        if nargout > 3
            varargout{2} = bestTilt;
        end
        if nargout > 4
            varargout{3} = peakValues;
        end
        
        return;
    else
        RevasMessage(['ReReference() is proceeding and overwriting an existing file. (' outputFilePath ')'], logBox);  
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
    
    if isfield(params,'position') && isfield(params,'timeSec')
        position = params.position;
        timeSec = params.timeSec;
    end
    
    if ~exist('position','var')
        error('ReReference: eye position array cannot be found!');   
    end

else % inputArgument is not a file path, but carries the eye position data.    
    position = inputArgument(:,1:size(inputArgument,2)-1);
    timeSec = inputArgument(:,size(inputArgument,2)); % last column is always 'time'
end


%% Handle local reference frame scenarios.

localRef = HandleInputArgument(params.referenceFrame);
if isempty(localRef)
    error('ReReference: error occured while loading local ref!');
end

%% Handle |globalRefArgument| scenarios.

globalRef = HandleInputArgument(params.globalRefArgument);
if isempty(globalRef)
    error('ReReference: error occured while loading global ref!');
end

%% Prepare

% get local ref size
[lHeight, lWidth] = size(localRef);

% set strip width to frame width, iff params.stripWidth is empty
if isempty(params.anchorStripWidth) || ~IsPositiveInteger(params.anchorStripWidth)
    params.anchorStripWidth = lWidth;
end
stripLeft = max(1,round((lWidth - params.anchorStripWidth)/2));
stripRight = min(lWidth,round((lWidth + params.anchorStripWidth)/2)-1);

% remove black edges around the refs, if any
localRef = PadNoise(localRef);
globalRef = PadNoise(globalRef);

% determine which part of localRef to use for re-referencing. We use
% params.anchorRowNumber if available. If not, we use the line contrast
% method: first, compute rms contrast of each scanline within localRef, and
% second, use the highest contrast region to extract anchoring strips.
if isempty(params.anchorRowNumber)
    
    % get a strip from the localRef based on local contrast
    lineContrast = medfilt1(std(double(localRef),[],2),31);
    [~, maxIx] = max(lineContrast);
    anchorSt = max(1,maxIx - params.anchorStripHeight); % intentionally avoided halving strip height since we are using central half below
    anchorEn = min(lHeight, maxIx + params.anchorStripHeight);
    params.anchorRowNumber = anchorSt;

else
    anchorSt = params.anchorRowNumber;
    anchorEn = anchorSt + params.anchorStripHeight - 1;
end
anchorStrip = localRef(anchorSt:anchorEn,stripLeft:stripRight);

% create a struct for full-reference crosscorr.
anchorOp = struct;
anchorOp.enableGPU = params.enableGPU;
anchorOp.corrMethod = params.corrMethod;
anchorOp.adaptiveSearch = false;
anchorOp.rowStart = 1;
anchorOp.rowEnd = size(globalRef,1);
anchorOp.referenceFrame = globalRef;

% if torsional search is enabled, rotate anchor strip before locating on
% glboal ref
if ~params.fixTorsion
    tilts = 0;
else
    tilts = params.tilts;
end

peakValues = zeros(size(tilts));

anchorStripSize = size(anchorStrip);
halfStripSize = round(anchorStripSize/2);

%% Find tilt



% locate strip on global ref
for i=1:length(tilts)
    if ~abortTriggered
    
        % rotate strip
        tempStrip = imrotate(anchorStrip, tilts(i),'bilinear');
        
        st = round((size(tempStrip) - halfStripSize)/2);
        
        thisStrip = tempStrip(st(1):st(1)+halfStripSize(1)-1, st(2):st(2)+halfStripSize(2)-1);

        % find peak values vs tilt
        [~,~,~, peakValues(i)] = LocateStrip(thisStrip,anchorOp); 
    end
end

% Interpolate for best tilt 
% if there is more than one tilt, try interpolating for best tilt and
% recompute the offset at that tilt. if not, simply pass the already
% computed values.
if length(tilts) > 1
    
    newTilts = min(tilts): 0.05 :max(tilts);
    newPeakValues = interp1(tilts, peakValues, newTilts, 'spline');

    % find best tilt
    [~, tiltIx] = max(newPeakValues);
    bestTilt = newTilts(tiltIx);
else
    bestTilt = tilts;

end

% rotate local ref with best tilt
correctedLocalRef = PadNoise(imrotate(localRef, bestTilt,'bilinear','crop'));


%% now re-reference

% get a strip from corrected localRef
rereferenceStrip = correctedLocalRef(anchorSt:anchorEn, stripLeft:stripRight);

% the location of the anchor strip in localRef
xPeakLocal = params.anchorStripWidth + stripLeft - 1;
yPeakLocal = anchorSt + size(anchorStrip,1) - 1;

offset = ReReferenceHelper(rereferenceStrip, anchorOp, anchorStripSize, [xPeakLocal yPeakLocal]);

% apply re-referencing
position = position + offset;


%% assign outputs
if writeResult
    outputArgument = outputFilePath;
else
    outputArgument = [position timeSec];
end


%% Plot stimuli on reference frame
if ~abortTriggered && params.enableVerbosity 
    
    finalLocalRef = imtranslate(correctedLocalRef,offset,'fillvalues',nan);
    padToSize = max([size(finalLocalRef); size(globalRef)],[],1);
    paddedGlobalRef = padarray(globalRef, padToSize-size(globalRef),nan,'post');
    axes(params.axesHandles(1));
    imshowpair(paddedGlobalRef,finalLocalRef);

end


%% Save filtered data.
if writeResult && ~abortTriggered
    
    % remove unnecessary fields
    params = RemoveFields(params,{'logBox','axesHandles','globalRefArgument',...
        'referenceFrame'}); 

    % save output
    save(outputFilePath,'position','timeSec','offset','bestTilt','params','peakValues');
end

%% return optional outputs

params.position = position;
params.timeSec = timeSec;
params.offset = offset;
params.bestTilt = bestTilt;

if nargout > 2
    varargout{1} = offset;
end
if nargout > 3
    varargout{2} = bestTilt;
end
if nargout > 4
    varargout{3} = peakValues;
end



function [offset, peakValue, peakLoc, cMap] = ...
    ReReferenceHelper(thisStrip, anchorOp, anchorStripSize, localPeakLoc)

% find the anchor strip in current reference frame
[cMap, xPeak, yPeak, peakValue] = LocateStrip(thisStrip,anchorOp,struct);  

% adjust for strip dimensions (will be different for each tilt)
xPeaksGlobal = xPeak - (size(thisStrip,2) - anchorStripSize(2));
yPeaksGlobal = yPeak - (size(thisStrip,1) - anchorStripSize(1));
peakLoc = [xPeaksGlobal yPeaksGlobal];

% offset needed to re-reference the strips from localRef to globalRef
offset = peakLoc - localPeakLoc + [1 0];



% add noise to zero-padded regions due to tilt
function output = PadNoise(input)

output = input;
padIndices = input == 0;
noisePixels = datasample(input(~padIndices), sum(padIndices(:)), 'replace',true);
output(padIndices) = noisePixels;


% Handle input arguments
function output = HandleInputArgument(inputArgument)

output = [];

if ischar(inputArgument) % inputArgument is a file path

    % check if input file exists
    if ~exist(inputArgument,'file')
        return;
    end
    
    try
        % load the inputArgument
        load(inputArgument,'referenceFrame');
        output = referenceFrame;
    catch
        try
            % maybe this is a field of params in this file
            load(inputArgument,'params');
            output = params.referenceFrame;
        catch
            
            try
                % maybe this is an image file
                im = imread(inputArgument);
            
                % if it's an rgb image, convert to grayscale
                if size(im,3) > 1
                    im = rgb2gray(im);
                end

                % if it's a double image, convert it to uint8
                if max(im(:)) < 1 &&  ~isa(im,'uint8')
                    output = uint8(im*255);
                else
                    output = im;
                end  
                
            catch handleErr
                RevasError('An error occurred in ReReference in HandleInputArgument.');
                handleErr.rethrow;
            end
            
        end
    end

else % inputArgument is not a file path, but carries the referenceFrame.
    if max(inputArgument(:)) < 1 &&  ~isa(inputArgument,'uint8')
        output = uint8(inputArgument*255);
    else
        output = inputArgument;
    end
end

