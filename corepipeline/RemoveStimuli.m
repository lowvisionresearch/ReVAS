function [outputVideo, varargout] = RemoveStimuli(inputVideo, params)
%REMOVE STIMULI Finds and removes stimuli from each frame. 
% Stimulus locations are saved in a mat file, and the video with stimuli
% removed is saved with "_nostim" suffix.
%
%   ----------------------------------- 
%   Input
%   ----------------------------------- 
%   |inputVideo| is either the path to the video, or the video matrix
%   itself. In the former situation, the result is stored in a new video
%   file with '_nostim' appended to the input video file name. In the
%   latter situation, no video is written and the result is returned. If
%   the input video is a path, stimulus locations are written to a mat file
%   with 'stimlocs' keyword in its name. Regardless of the input, stimulus
%   locations file path and array are returned at the 2nd and 3rd output 
%   arguments.
%
%   |params| is a struct as specified below.
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is either the path to the video, or the video matrix
%   itself after stimulus removal. 
%
%   varargout{1} = matFilePath
%   varargout{2} = stimulus locations.
%
%
%   -----------------------------------
%   Fields of the |params| 
%   -----------------------------------
%   overwrite          : set to true to overwrite existing files.
%                        Set to false to abort the function call if the
%                        files already exist. (default false)
%   enableVerbosity    : set to true to report back plots during execution.
%                        (default false)
%   axesHandles        : axes handle for giving feedback. if not
%                        provided or empty, new figures are created.
%                        (relevant only when enableVerbosity is true)
%   stimulus           : is a path to a stimulus image, a 2D or 3D (rgb) array, 
%                        or a struct containing a |size| field which is the
%                        size of the stimulus in pixels (default 11), and a
%                        |thickness| field which is the thickness of the
%                        default cross shape in pixels (default 1). (default
%                        is dynamically generated stimulus with
%                        aforementioned defaults)
%   removalAreaSize    : is the size of the rectangle to remove from the
%                        video, centered around the identified stimulus
%                        location. The format is [width length], given in
%                        pixels. (default [11 11]). This option is useful
%                        when one wants to use the small white cross to
%                        remove stimulus but include some region around it,
%                        without needing to specify the exact shape of the
%                        stimulus. if empty, ignored completely.
%   minimumPeakThreshold:the minimum value above which a peak
%                        needs to be in order to be considered 
%                        a valid correlation. (this applies
%                        regardless of enableGaussianFiltering)
%                        (default 0)
%   badFrames          : specifies blink/bad frames. we can skip those but
%                        we need to make sure to keep a record of 
%                        discarded frames. 
%   fillingMethod      : 'noise' for replacing stimulus with gaussian noise
%                        using image statistics. 'resample' for randomly
%                        sampling pixels from the frame itself (produces
%                        better filling in).
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'tslo.avi';
%       params.enableVerbosity = true;
%       params.overwrite = true;
%       params.stimulus = struct;
%       params.stimulus.size = 11;
%       params.stimulus.thickness = 1;
%       params.stimulus.polarity = 1;
%   OR
%       params.stimulus = <path to an image>;
%       params.removalAreaSize = [11 11];
%       RemoveStimuli(inputVideo, params);

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

if ~isfield(params, 'overwrite')
    overwrite = false; 
else
    overwrite = params.overwrite;
end

if ~isfield(params, 'enableVerbosity')
    enableVerbosity = false; 
else
    enableVerbosity = params.enableVerbosity;
end

if ~isfield(params, 'axesHandles')
    axesHandles = nan; 
else
    axesHandles = params.axesHandles;
end

if ~isfield(params, 'minimumPeakThreshold')
    minimumPeakThreshold = 0.6;
    RevasWarning(['RemoveStimuli is using default parameter for minimumPeakThreshold: ' num2str(minimumPeakThreshold)] , params);
else
    minimumPeakThreshold = params.minimumPeakThreshold;
end

if ~isfield(params, 'frameRate')
    frameRate = 30;
    RevasWarning(['RemoveStimuli is using default parameter for frameRate: ' num2str(frameRate)] , params);
else
    frameRate = params.frameRate;
end

if ~isfield(params, 'fillingMethod')
    fillingMethod = 'resample';
    RevasWarning(['RemoveStimuli is using default parameter for fillingMethod: ' fillingMethod] , params);
else
    fillingMethod = params.fillingMethod;
end

if ~isfield(params, 'badFrames')
    badFrames = false;
    RevasWarning('RemoveStimuli is using default parameter for badFrames: none.', params);
else
    badFrames = params.badFrames;
end

if ~isfield(params, 'stimulus')
    stimulus.size = 11;
    stimulus.thickness = 1;
    stimulus.polarity = 1;
    description = 'Cross: 11px size, 1px thickness, positive polarity'; 
    RevasWarning(['RemoveStimuli is using default parameter for stimulus: ' description], params);
else
    % Two stimulus input types are acceptable:
    % - Path to an image of the stimulus
    % - A struct describing |size| and |thickness| of stimulus

    stimulus = params.stimulus;
    
    % if it's a path, read the image from the path
    if ischar(stimulus) 
        stimulusMatrix = imread(stimulus);
    else
        stimulusMatrix = stimulus;
    end

    % if it's a numeric array, check if it's RGB
    if isnumeric(stimulusMatrix)
        [~, ~, numChannels] = size(stimulusMatrix);
        if numChannels == 3
            stimulusMatrix = rgb2gray(stimulusMatrix);
        end
    end
end

% if stimulus is a struct, construct the matrix assuming that stimulus is a
% cross (more like a plus sign)
if isstruct(stimulus)
    stimulusMatrix = MakeStimulusCross(stimulus.size, stimulus.thickness, stimulus.polarity);
    
    % zeropad for improved matching -- important for making this step
    % robust against being used for videos with no stimulus.
    stimulusMatrix = padarray(stimulusMatrix, stimulus.size*[1 1],~(stimulus.polarity),'both');
end

% make sure to conver stimulus to uint8
stimulusMatrix = uint8(stimulusMatrix);

if ~isfield(params, 'removalAreaSize')
    removalAreaSize = []; % pixels
    RevasWarning(['RemoveStimuli is using default parameter for removalAreaSize: ' num2str(removalAreaSize)] , params);
else
    removalAreaSize = params.removalAreaSize;
    if isscalar(removalAreaSize)
        removalAreaSize = removalAreaSize*[1 1];
    end
end

%% Handle overwrite scenarios.

if writeResult
    outputVideoPath = Filename(inputVideo, 'removestim');
    matFilePath = Filename(inputVideo, 'stimlocs');
    if nargout > 1
        varargout{1} = matFilePath;
    end
    
    if ~exist(matFilePath, 'file') && ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~overwrite
        
        % if file exists and overwrite is set to false, then read the file
        % contents and return that.
        load(matFilePath,'stimulusLocations');
        if nargout > 2
            varargout{2} = stimulusLocations;
        end
        
        RevasWarning('RemoveStimuli() did not execute because it would overwrite existing file.', params);
        return;
    else
        RevasWarning('RemoveStimuli() is proceeding and overwriting an existing file.', params);
    end
end



%% Allow for aborting if not parallel processing
global abortTriggered;

% parfor does not support global variables.
% cannot abort when run in parallel.
if isempty(abortTriggered)
    abortTriggered = false;
end

%% Create reader/writer objects and get some info on videos

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    % Determine dimensions of video.
    reader = VideoReader(inputVideo);
    frameRate = reader.FrameRate;
    width = reader.Width;
    height = reader.Height;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    
    % Determine dimensions of video.
    [height, width, numberOfFrames] = size(inputVideo);
    
    % preallocate the output video array
    outputVideo = zeros(height, width, numberOfFrames-sum(badFrames),'uint8');
end

%% badFrames handling
% If badFrames is not provided, use all frames
if length(badFrames)<=1 && ~badFrames
    badFrames = false(numberOfFrames,1);
end

% If badFrames are provided but its size don't match the number of frames
if length(badFrames) ~= numberOfFrames
    badFrames = false(numberOfFrames,1);
    RevasWarning('TrimVideo(): size mismatch between ''badFrames'' and input video. Using all frames for this video.', params);  
end


%% Some preparation/preallocation before for-loop

% Populate time array
timeArray = (1:numberOfFrames)' / frameRate;   

% preallocate two columns for horizontal and vertical movements
rawStimulusLocations = nan(numberOfFrames, 2);
stimulusLocations = nan(numberOfFrames, 2);
peakValues = nan(numberOfFrames, 1);

sw = size(stimulusMatrix,2);
sh = size(stimulusMatrix,1);

if isempty(removalAreaSize)
    % look at the values in stimulus. if more zeros, than it's most
    % probably a positive polarity target (stimulus is brighter). If not,
    % it's a black target.
    nCounts = histcounts(stimulusMatrix(:),2);
    if diff(nCounts) > 0
        indices = find(~stimulusMatrix);
    else
        indices = find(stimulusMatrix);
    end
    halfWidth = floor(sw/2);
    halfHeight = floor(sh/2);
else
    halfWidth = floor(removalAreaSize(1)/2);
    halfHeight = floor(removalAreaSize(2)/2);
end


%% Find stimulus locations
    
for fr = 1:numberOfFrames
    if ~abortTriggered
        
        % get next frame
        if writeResult
            frame = readFrame(reader);
            if ndims(frame) == 3
                frame = rgb2gray(frame);
            end
        else
            frame = inputVideo(:,:, fr);
        end

        % if it's a blink frame, skip it.
        if badFrames(fr)
            continue;
        end

        % locate the stimulus
        [correlationMap,xPeak,yPeak,peakValues(fr)] = matchTemplateOCV(stimulusMatrix, frame); %#ok<ASGLU>

        % update stimulus locations
        rawStimulusLocations(fr,:) = [xPeak yPeak] - floor([sw sh]/2);

        % assess quality, if satisfied use the sample. otherwise discard.
        if peakValues(fr) >= minimumPeakThreshold
            stimulusLocations(fr,:) = rawStimulusLocations(fr,:);

            % Remove target / replace it with noise here
            if all(~isnan(stimulusLocations(fr,:)))
                
                % if additional area to be removed
                xSt = max(1,stimulusLocations(fr,1) - halfWidth);
                xEn = min(width, stimulusLocations(fr,1) + halfWidth);
                ySt = max(1,stimulusLocations(fr,2) - halfHeight);
                yEn = min(height, stimulusLocations(fr,2) + halfHeight); 

                % fill in the stimulus area with either noise or resampled
                % pixels
                switch fillingMethod
                    case 'noise'
                        % compute image stats
                        stdev = rms(frame(:));
                        md = double(median(frame(:)));
                        
                        % Generate noise
                        if isempty(removalAreaSize)
                            subFrame = frame(ySt:yEn, xSt:xEn);
                            subFrame(indices) = randn(length(indices),1) * stdev + 1.2*md;
                        else
                            % Adjust to the mean and sd of current frame
                            subFrame = randn(yEn-ySt+1,xEn-xSt+1) * stdev + 1.2*md;
                        end
                    case 'resample'
                        % Generate noise
                        if isempty(removalAreaSize)
                            subFrame = frame(ySt:yEn, xSt:xEn);
                            subFrame(indices) = frame(randi(height*width,length(indices),1));
                        else
                            % Adjust to the mean and sd of current frame
                            subFrame = frame(randi(height*width,(yEn-ySt+1),(xEn-xSt+1)));
                        end
                    otherwise 
                end


                % Put back the filled-in region into the frame
                frame(ySt:yEn, xSt:xEn) = uint8(subFrame);
            end
        end

        % write out
        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~badFrames(1:fr));
            outputVideo(:, :, nextFrameNumber) = frame; 
        end


        % Show the filled frame if verbosity enabled
        if enableVerbosity
            if all(ishandle(axesHandles))
                axes(axesHandles(1)); %#ok<LAXES>
                colormap(axesHandles(1), 'default');
            else
                figure(432);
            end
            cla;
            imagesc(frame); axis image; hold on;
            title([num2str(fr) ' out of ' num2str(numberOfFrames)]);
            colormap(gray(256));
            colorbar;
            caxis([0 255]);
        end
    end
end % end of video


%% return results, close up objects

if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if abortTriggered
        delete(outputVideoPath)
    end
else
    outputVideo = inputVideo;
end


%% if requested, return the stimulus locations as output

if nargout > 2 
    varargout{2} = stimulusLocations;
end

%% Save to output mat file

if writeResult
    save(matFilePath, 'stimulusLocations', 'stimulus', 'peakValues',...
        'rawStimulusLocations');
end

%% if verbosity enabled, show the extracted stimulus locations

if enableVerbosity
    % Plotting bottom right corner of box surrounding stimulus.
    if all(ishandle(axesHandles))
        axes(axesHandles(2)); 
        colormap(axesHandles(2), 'default');
    else
        figure(123);
    end
    cla;
    plot(timeArray, stimulusLocations,'-','linewidth',2); hold on;
    title('Stimulus Locations');
    xlabel('Time (sec)');
    ylabel('Stimulus Locations (pixels)');
    legend('show');
    legend('Horizontal', 'Vertical');
    set(gca,'fontsize',14);
    grid on;
    drawnow;
end


