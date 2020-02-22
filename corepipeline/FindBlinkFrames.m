function [inputVideo, params, varargout] = FindBlinkFrames(inputVideo, params)
%FIND BLINK FRAMES  Records in a mat file the frames in which a blink
%                   occurred. Blinks are considered to be frames in which
%                   the frame's mean and standard deviation are reduced
%                   whereas skewness and kurtosis of its histogram are
%                   increased. We use k-means clustering to find bad frames
%                   using these four image stats. There is a hard threshold
%                   for separating the two clusters, i.e., if mean values
%                   of bad frames and good frames are no different than
%                   this threshold, we take it as absence of any blinks/bad
%                   frames.
%
%   The result is stored with '_blinkframes' appended to the input video
%   file name if a |inputVideo| path is provided, and it is also returned
%   by the function. If the input video is a path, blink frames are written
%   to a file. Regardless of the input, an index array of |badFrames| is
%   returned at the first output argument.
%
%   -----------------------------------
%   Input
%   -----------------------------------
%   |inputVideo| is the path to the video or a matrix representation of the
%   video that is already loaded into memory.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite           :   set to 1 to overwrite existing files resulting 
%                           from calling FindBlinkFrames.
%                           Set to 0 to params.abort the function call if the
%                           files exist in the current directory.
%   enableVerbosity     :   set to true to report back plots during execution.
%                           (default false)
%   stitchCriteria      :   optional--specify the maximum distance (in frames)
%                           between blinks, below which two blinks will be
%                           marked as one. For example, if badFrames is 
%                           [8, 9, 11, 12], this represents two blinks, one
%                           at frames 8 and 9, and the other at frames 
%                           11 and 12. If stitchCriteria is 2, then 
%                           badFrames becomes [8, 9, 10, 11, 12] because the
%                           distance between the blinks [8, 9] and [11, 12]
%                           are separated by only one frame, which is less
%                           than the specified stitch criteria.
%   numberOfBins        :   optional--specify number of bins for image 
%                           histogram. All image stats are computed using
%                           this histogram. The default value is 256.
%   meanDifferenceThreshold: minimum mean gray level distance between two
%                           clusters representing bad and good frames.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |inputVideo| is directly passed from input argument.
%
%   |params| structure with badFrames as a new field
%
%   varargout{1} = badFramesMatFilePath
%   varargout{2} = image statistics extracted from all frames.
%   varargout{3} = candidate for initial reference frame based on means.
%
%                          
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       videoPath = 'aoslo-blink.avi';
%       params.overwrite = true;
%       params.stitchCriteria = 1;
%       params.numberOfBins = 256;
%       params.meanDifferenceThreshold = 256;
%       FindBlinkFrames(videoPath, params);


%% Determine inputVideo type.
if ischar(inputVideo)
    % A path was passed in.
    % Read the video and once finished with this module, write the result.
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


%% Handle GUI mode
% params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
    isGUI = true;
else
    logBox = [];
    isGUI = false;
end

% params will have access to a uicontrol object in GUI mode. so if it does
% not already have that, create the field and set it to false so that this
% module can be used without the GUI
if ~isfield(params,'abort')
    params.abort.Value = false;
end


%% Handle verbosity 
% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Find Blink Frames',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.67 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');
    params.axesHandles(1) = subplot(1,1,1);
end

if params.enableVerbosity
    cla(params.axesHandles(1));
    tb = get(params.axesHandles(1),'toolbar');
    tb.Visible = 'on';
end


%% Handle overwrite scenarios.
if writeResult
    badFramesMatFilePath = [inputVideo(1:end-4) '_blinkframes.mat'];
    params.badFramesMatFilePath = badFramesMatFilePath;
    if nargout > 1 
        varargout{1} = badFramesMatFilePath;
    end
    
    if ~exist(badFramesMatFilePath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~params.overwrite
        load(badFramesMatFilePath,'badFrames','imStats','initialRef');
        params.badFrames = badFrames;
        if nargout > 2
            varargout{2} = imStats;
        end
        if nargout > 3
            varargout{3} = initialRef;
        end
        RevasWarning(['FindBadFrames() did not execute because it would overwrite existing file. (' badFramesMatFilePath ')'], logBox);
        return;
    else
        RevasWarning(['FindBadFrames() is proceeding and overwriting an existing file. (' badFramesMatFilePath ')'], logBox);
    end
end





%% Read in video frames and collect some image statistics

if writeResult
    reader = VideoReader(inputVideo);
    numberOfFrames = reader.FrameRate * reader.Duration;
else
    [~, ~, numberOfFrames] = size(inputVideo);
end

means = zeros(numberOfFrames,1);
stds = zeros(numberOfFrames,1);
skews = zeros(numberOfFrames,1);
kurtoses = zeros(numberOfFrames,1);

% go over frames and compute image stats for each frame
for fr = 1:numberOfFrames
    if ~params.abort.Value
        % get next frame
        if writeResult
            % handle rgb frames
            frame = readFrame(reader);
            if ndims(frame) == 3
                frame = rgb2gray(frame);
            end
        else
            frame = inputVideo(:, :, fr);
        end

        % compute image histogram
        [counts, bins] = imhist(frame, params.numberOfBins);

        % compute image stats from histogram
        numOfPixels = sum(counts);
        means(fr) = sum(bins .* counts) / numOfPixels;
        stds(fr) = sqrt(sum((bins - means(fr)) .^ 2 .* counts) / (numOfPixels-1));
        skews(fr) = sum((bins - means(fr)) .^ 3 .* counts) / ((numOfPixels - 1) * stds(fr)^3);
        kurtoses(fr) = sum((bins - means(fr)) .^ 4 .* counts) / ((numOfPixels - 1) * stds(fr)^4);
    else
        break;
    end
    
    if isGUI
        pause(.001);
    end
end


%% Identify bad frames

if ~params.abort.Value
    % use all image stats to detect 2 clusters
    [idx, centroids] = kmeans([means stds skews kurtoses],2);

    % if centroids are too close, no blinks found.
    if abs(diff(centroids(:,1))) < params.meanDifferenceThreshold
        badFrames = false(numberOfFrames,1);
    else
        % select the cluster with smaller mean as the bad frames
        [~,whichClusterRepresentsBadFrames] = min(centroids(:,2));
        badFrames = idx == whichClusterRepresentsBadFrames;

        % Lump together blinks that are < |stitchCriteria| frames apart
        [st, en] = GetOnsetOffset(badFrames);
        [st, en] = MergeEvents(st, en, params.stitchCriteria);

        % note that we keep badFrames in a logical array format to preserve the
        % length of the original video.
        badFrames = GetIndicesFromOnsetOffset(st,en,numberOfFrames);
    end

    % include one frame before and one frame after to really make sure we don't
    % waste time in crappy frames
    badFrames = conv(badFrames,[1 1 1],'same') ~= 0;

    % include badFrames as a field under params structure to be used in
    % successive stages in the pipeline
    params.badFrames = badFrames;
end

%% Return image stats if user asks for it or results are to be written to a file

if nargout > 2 || writeResult
    imStats = struct;
    imStats.means = means;
    imStats.stds = stds;
    imStats.skews = skews;
    imStats.kurtoses = kurtoses;
    varargout{2} = imStats;
end


%% If user asks for it or results are to be written to a file, get a candidate frame for initial ref

if nargout > 3 || writeResult
    [~,initialRef] = max(means);
    varargout{3} = initialRef;
end



%% if verbosity enabled, show the found blink frames

if params.enableVerbosity && ~params.abort.Value
    
    badFrameNumbers = find(badFrames);
    p = [];
    mSize = 40;
    lt = 1.5;
    p(1) = plot(params.axesHandles(1),means,'-','linewidth',lt); 
    hold(params.axesHandles(1),'on');
    p(2) = plot(params.axesHandles(1),stds,'-','linewidth',lt);
    p(3) = plot(params.axesHandles(1),skews,'-','linewidth',lt);
    p(4) = plot(params.axesHandles(1),kurtoses,'-','linewidth',lt); 
    
    scatter(params.axesHandles(1),badFrameNumbers, means(badFrames),mSize,get(p(1),'color'),'filled'); 
    scatter(params.axesHandles(1),badFrameNumbers, stds(badFrames),mSize,get(p(2),'color'),'filled'); 
    scatter(params.axesHandles(1),badFrameNumbers, skews(badFrames),mSize,get(p(3),'color'),'filled'); 
    scatter(params.axesHandles(1),badFrameNumbers, kurtoses(badFrames),mSize,get(p(4),'color'),'filled'); 
    
    xlabel(params.axesHandles(1),'Frame number');
    ylabel(params.axesHandles(1),'Image stats');
    legend(params.axesHandles(1),{'means', 'stds','skews','kurtoses','badFrames'});
    set(params.axesHandles(1),'fontsize',10);
    grid(params.axesHandles(1),'on');
    drawnow;
end



%% Save to output mat file

% remove unnecessary fields
abort = params.abort.Value;
params = RemoveFields(params,{'logBox','axesHandles','abort'}); 

if writeResult && ~abort
    % save results
    save(badFramesMatFilePath, 'badFrames','imStats','initialRef','params');
end
