function [outputVideo, params, varargout] = RemoveStimuli(inputVideo, params)
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
%   Fields of the |params| 
%   -----------------------------------
%   overwrite          : set to true to overwrite existing files.
%                        Set to false to params.abort the function call if the
%                        files already exist. (default false)
%   enableVerbosity    : set to true to report back plots during execution.
%                        (default false)
%   axesHandles        : axes handle for giving feedback. if not
%                        provided or empty, new figures are created.
%                        (relevant only when enableVerbosity is true)
%   stimulus           : is a path to a stimulus image, a 2D or 3D (rgb) array, 
%                        or empty. 
%   stimulusSize       : size of the stimulus in pixels (default 11), 
%   stimulusThickness  : thickness of the default cross shape in pixels 
%                        (default 1). 
%   stimulusPolarity   : 1 or 0, or true or false. If true or 1, stimulus
%                        is a white cross on a black background.
%   removalAreaSize    : is the size of the rectangle to remove from the
%                        video, centered around the identified stimulus
%                        location. The format is [width length], given in
%                        pixels. (default [11 11]). This option is useful
%                        when one wants to use the small white cross to
%                        remove stimulus but include some region around it,
%                        without needing to specify the exact shape of the
%                        stimulus. if empty, ignored completely.
%   minPeakThreshold   :the minimum value above which a peak
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
%
%   -----------------------------------
%   Output
%   -----------------------------------
%   |outputVideo| is either the path to the video, or the video matrix
%   itself after stimulus removal. 
%
%   |params| structure.
%
%   varargout{1} = matFilePath
%   varargout{2} = stimulus locations.
%
%   -----------------------------------
%   Example usage
%   -----------------------------------
%       inputVideo = 'tslo.avi';
%       params.enableVerbosity = true;
%       params.overwrite = true;
%       params.stimulus = [];
%       params.stimuluSize = 11;
%       params.stimuluThickness = 1;
%       params.stimuluPolarity = 1;
%   OR
%       params.stimulus = <path to an image>;
%       params.removalAreaSize = [11 11];
%       RemoveStimuli(inputVideo, params);



%% in GUI mode, params can have a field called 'logBox' to show messages/warnings 
if isfield(params,'logBox')
    logBox = params.logBox;
else
    logBox = [];
end


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

%% Handle verbosity 

% check if axes handles are provided, if not, create axes.
if params.enableVerbosity && isempty(params.axesHandles)
    fh = figure(2020);
    set(fh,'name','Remove Stimuli',...
           'units','normalized',...
           'outerposition',[0.16 0.053 0.67 0.51],...
           'menubar','none',...
           'toolbar','none',...
           'numbertitle','off');
    params.axesHandles(1) = subplot(2,3,[1 2 4 5]);
    params.axesHandles(2) = subplot(2,3,3);
    params.axesHandles(3) = subplot(2,3,6);
    
end

if params.enableVerbosity
    for i=1:3
        cla(params.axesHandles(i));
        tb = get(params.axesHandles(i),'toolbar');
        tb.Visible = 'on';
    end
end


%% Handle overwrite scenarios.

if writeResult
    outputVideoPath = Filename(inputVideo, 'removestim');
    matFilePath = Filename(inputVideo, 'stimlocs');
    params.outputVideoPath = outputVideoPath;
    params.matFilePath = matFilePath;
    
    if nargout > 2
        varargout{1} = matFilePath;
    end
    
    if ~exist(matFilePath, 'file') && ~exist(outputVideoPath, 'file')
        % left blank to continue without issuing warning in this case
    elseif ~params.overwrite
        
        % if file exists and overwrite is set to false, then read the file
        % contents and return that.
        load(matFilePath,'stimulusLocations','params');
        params.stimulusLocations = stimulusLocations;
        if nargout > 3
            varargout{2} = stimulusLocations;
        end
        
        RevasWarning('RemoveStimuli() did not execute because it would overwrite existing file.', logBox);
        return;
    else
        RevasWarning('RemoveStimuli() is proceeding and overwriting an existing file.', logBox);
    end
else
    matFilePath = [];
end

%% Handle stimulus separately

% if it's a path, read the image from the path
if ischar(params.stimulus) 
    stimulusMatrix = imread(params.stimulus);
else
    stimulusMatrix = params.stimulus;
end

% if it's a numeric array, check if it's RGB
if isnumeric(stimulusMatrix)
    numChannels = size(stimulusMatrix,3);
    if numChannels == 3
        stimulusMatrix = rgb2gray(stimulusMatrix);
    end
end


% if stimulus is empty, construct the matrix assuming that stimulus is a
% cross (more like a plus sign)
if isempty(params.stimulus)
    stimulusMatrix = MakeStimulusCross(params.stimulusSize, params.stimulusThickness, params.stimulusPolarity);
    
    % zeropad for improved matching -- important for making this step
    % robust against being used for videos with no stimulus.
    stimulusMatrix = padarray(stimulusMatrix, params.stimulusSize*[1 1],~(params.stimulusPolarity),'both');
end

% make sure to conver stimulus to uint8
stimulusMatrix = uint8(stimulusMatrix);

% at this point, stimulusMatrix should be a 2D array.
assert(ismatrix(stimulusMatrix));


%% Create reader/writer objects and get some info on videos

if writeResult
    writer = VideoWriter(outputVideoPath, 'Grayscale AVI');
    open(writer);

    % Determine dimensions of video.
    reader = VideoReader(inputVideo);
    params.frameRate = reader.FrameRate;
    width = reader.Width;
    height = reader.Height;
    numberOfFrames = reader.FrameRate * reader.Duration;
    
else
    
    % Determine dimensions of video.
    [height, width, numberOfFrames] = size(inputVideo);
    
    % preallocate the output video array
    outputVideo = zeros(height, width, numberOfFrames-sum(params.badFrames),'uint8');
end

%% badFrames handling
params = HandleBadFrames(numberOfFrames, params, callerStr);


%% Some preparation/preallocation before for-loop

% Populate time array
if length(params.badFrames) > numberOfFrames
    timeSec = (find(~params.badFrames)-1)' / params.frameRate;   
else
    timeSec = (0:(numberOfFrames-1))' / params.frameRate;   
end

% preallocate two columns for horizontal and vertical movements
rawStimulusLocations = nan(numberOfFrames, 2);
stimulusLocations = nan(numberOfFrames, 2);
peakValues = nan(numberOfFrames, 1);

sw = size(stimulusMatrix,2);
sh = size(stimulusMatrix,1);

if isempty(params.removalAreaSize)
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
    halfWidth = floor(params.removalAreaSize(1)/2);
    halfHeight = floor(params.removalAreaSize(2)/2);
end

isFirstTimePlotting = true;

%% Find stimulus locations

isGUI = isfield(params,'logBox');
    
for fr = 1:numberOfFrames
    if ~params.abort.Value
        
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
        if params.skipFrame(fr)
            continue;
        end

        % locate the stimulus
        [correlationMap,xPeak,yPeak,peakValues(fr)] = matchTemplateOCV(stimulusMatrix, frame); %#ok<ASGLU>

        % update stimulus locations
        rawStimulusLocations(fr,:) = [xPeak yPeak] - floor([sw sh]/2);

        % assess quality, if satisfied use the sample. otherwise discard.
        if peakValues(fr) >= params.minPeakThreshold
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
                switch params.fillingMethod
                    case 'noise'
                        % compute image stats
                        stdev = rms(frame(:));
                        md = double(median(frame(:)));
                        
                        % Generate noise
                        if isempty(params.removalAreaSize)
                            subFrame = frame(ySt:yEn, xSt:xEn);
                            subFrame(indices) = randn(length(indices),1) * stdev + 1.2*md;
                        else
                            % Adjust to the mean and sd of current frame
                            subFrame = randn(yEn-ySt+1,xEn-xSt+1) * stdev + 1.2*md;
                        end
                    case 'resample'
                        % Generate noise
                        if isempty(params.removalAreaSize)
                            subFrame = frame(ySt:yEn, xSt:xEn);
                            subFrame(indices) = frame(randi(height*width,length(indices),1));
                        else
                            % Adjust to the mean and sd of current frame
                            subFrame = frame(randi(height*width,(yEn-ySt+1),(xEn-xSt+1)));
                        end
                    otherwise 
                        error('RemoveStimuli: unknown filling method');
                end


                % Put back the filled-in region into the frame
                frame(ySt:yEn, xSt:xEn) = uint8(subFrame);
            end
        end

        % write out
        if writeResult
            writeVideo(writer, frame);
        else
            nextFrameNumber = sum(~params.badFrames(1:fr));
            outputVideo(:, :, nextFrameNumber) = frame; 
        end


        % visualization, if requested.
        if params.enableVerbosity > 1

            if isFirstTimePlotting
                
                % show cross-correlation output
                axes(params.axesHandles(1)); %#ok<LAXES>
                im = imagesc(frame);
                axis(params.axesHandles(1),'image');
                title(params.axesHandles(1),[num2str(fr) ' out of ' num2str(numberOfFrames)]);
                colormap(params.axesHandles(1),gray(256));
                caxis(params.axesHandles(1),[0 255]);

                % show peak values=
                p21 = plot(params.axesHandles(2),timeSec,peakValues,'-','linewidth',2); 
                hold(params.axesHandles(2),'on');
                plot(params.axesHandles(2),timeSec([1 end]),params.minPeakThreshold*ones(1,2),'--','color',.7*[1 1 1],'linewidth',2);
                set(params.axesHandles(2),'fontsize',14);
                xlabel(params.axesHandles(2),'time (sec)');
                ylabel(params.axesHandles(2),'peak value');
                ylim(params.axesHandles(2),[0 1]);
                xlim(params.axesHandles(2),[0 max(timeSec)]);
                hold(params.axesHandles(2),'off');
                grid(params.axesHandles(2),'on');


                % show raw output traces
                p31 = plot(params.axesHandles(3),timeSec,rawStimulusLocations,'-','linewidth',2);
                set(params.axesHandles(3),'fontsize',14);
                xlabel(params.axesHandles(3),'time (sec)');
                ylabel(params.axesHandles(3),'stimulus location (px)');
                legend(params.axesHandles(3),{'hor','ver'});
                yMin = max([1, prctile(stimulusLocations,5,'all')-10]);
                yMax = min([max([height width]), prctile(stimulusLocations,95,'all')+10]);
                ylim(params.axesHandles(3),[yMin yMax]);
                xlim(params.axesHandles(3),[0 max(timeSec)]);
                hold(params.axesHandles(3),'off');
                grid(params.axesHandles(3),'on');
                
                isFirstTimePlotting = false;
            else
                
                im.CData = frame;
                set(p21,'YData',peakValues);
                set(p31(1),'YData',rawStimulusLocations(:,1));
                set(p31(2),'YData',rawStimulusLocations(:,2));
            end
            drawnow; % maybe needed in GUI mode.
        end
    else
        break;
    end % end of params.abort
    
    if isGUI
        pause(.001);
    end
end % end of video

%% return results, close up objects

if writeResult
    outputVideo = outputVideoPath;
    
    close(writer);
    
    % if aborted midway through video, delete the partial video.
    if params.abort.Value
        delete(outputVideoPath)
    end
else
    outputVideo = inputVideo;
end


%% handle variable output arguments

if nargout > 2 
    varargout{1} = matFilePath;
end
if nargout > 3 
    varargout{2} = stimulusLocations;
end


%% if verbosity enabled, show the extracted stimulus locations

if ~params.abort.Value && params.enableVerbosity
    
    % show cross-correlation output
    axes(params.axesHandles(1)); 
    imagesc(frame);
    axis(params.axesHandles(1),'image');
    title(params.axesHandles(1),[num2str(fr) ' out of ' num2str(numberOfFrames)]);
    colormap(params.axesHandles(1),gray(256));
    caxis(params.axesHandles(1),[0 255]);
    title(params.axesHandles(1),'stimulus removal example');

    % show peak values
    plot(params.axesHandles(2),timeSec,peakValues,'-','linewidth',2); 
    hold(params.axesHandles(2),'on');
    plot(params.axesHandles(2),timeSec([1 end]),params.minPeakThreshold*ones(1,2),'--','color',.7*[1 1 1],'linewidth',2);
    set(params.axesHandles(2),'fontsize',10);
    xlabel(params.axesHandles(2),'time (sec)');
    ylabel(params.axesHandles(2),'peak value');
    ylim(params.axesHandles(2),[0 1]);
    xlim(params.axesHandles(2),[0 max(timeSec)]);
    hold(params.axesHandles(2),'off');
    grid(params.axesHandles(2),'on');
    
    % show useful stimulus locations traces
    plot(params.axesHandles(3),timeSec,stimulusLocations,'-','linewidth',2);
    set(params.axesHandles(3),'fontsize',10);
    xlabel(params.axesHandles(3),'time (sec)');
    ylabel(params.axesHandles(3),'stimulus location (px)');
    legend(params.axesHandles(3),{'hor','ver'});
    yMin = max([1, prctile(stimulusLocations,5,'all')-10]);
    yMax = min([max([height width]), prctile(stimulusLocations,95,'all')+10]);
    ylim(params.axesHandles(3),[yMin yMax]);
    xlim(params.axesHandles(3),[0 max(timeSec)]);
    hold(params.axesHandles(3),'off');
    grid(params.axesHandles(3),'on');
end



%% Save to output mat file
if writeResult
    
    % remove unnecessary fields
    params = RemoveFields(params,{'logBox','axesHandles','abort'}); 
    
    save(matFilePath, 'stimulusLocations', 'params', 'peakValues',...
        'rawStimulusLocations','timeSec');
end
params.stimulusLocations = stimulusLocations;

