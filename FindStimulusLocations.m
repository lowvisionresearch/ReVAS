function FindStimulusLocations(inputVideoPath, stimulus, parametersStructure)
%FIND STIMULUS LOCATIONS Records in a mat file the location of the stimulus
%in each frame of the video.
%   The result is stored with '_stimlocs' appended to the input video file
%   name (and the file type is now .mat).
%
%   The mean and standard deviation of the pixels in each frame are also
%   saved as two separate arrays in this output mat file. 
%   |parametersStructure.overwrite| determines whether an existing output
%   file should be overwritten and replaced if it already exists.

outputFileName = [inputVideoPath(1:end-4) '_stimlocs'];

%% Handle overwrite scenarios.
if ~exist([outputFileName '.mat'], 'file')
    % left blank to continue without issuing warning in this case
elseif ~isfield(parametersStructure, 'overwrite') || ~parametersStructure.overwrite
    warning('FindStimulusLocations() did not execute because it would overwrite existing file.');
    return;
else
    warning('FindStimulusLocations() is proceeding and overwriting an existing file.');
end

%% Convert stimulus to matrix
% Two stimulus input types are acceptable:
% - Path to an image of the stimulus
% - A struct describing |size| and |thickness| of stimulus

if ischar(stimulus)
    % Read image from the path
    stimulus = imread(stimulus);
elseif isstruct(stimulus)
    % Both size and thickness must be odd
    if ~mod(stimulus.size, 2) == 1 || ~mod(stimulus.thickness, 2) == 1
        error('stimulus.size and stimulus.thickness must be odd');
    elseif stimulus.size < stimulus.thickness
        error('stimulus.size must be greater than stimulus.thickness');
    end
    % Draw the stimulus cross based on struct parameters provided.
    stimulusMatrix = ones(stimulus.size);
    stimulusMatrix(1:(stimulus.size-stimulus.thickness)/2, ...
        1:(stimulus.size-stimulus.thickness)/2) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end, ...
        1:(stimulus.size-stimulus.thickness)/2) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(1:(stimulus.size-stimulus.thickness)/2, ...
        stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulusMatrix(stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end, ...
        stimulus.size-(stimulus.size-stimulus.thickness)/2+1:end) = ...
        zeros((stimulus.size-stimulus.thickness)/2);
    stimulus = stimulusMatrix;
end

%% Find stimulus location of each frame

[videoInputArray, videoFrameRate] = VideoPathToArray(inputVideoPath);

frameHeight = size(videoInputArray, 1);
frameWidth = size(videoInputArray, 2);
numberOfFrames = size(videoInputArray, 3);
samplingRate = videoFrameRate;

[videoInput, ~] = VideoPathToArray(inputVideoPath);

% Populate time array
timeArray = (1:numberOfFrames)' / samplingRate;

% Call normxcorr2() to find stimulus on each frame
% Note that calculation for each array value does not end with this loop,
% the logic below the loop in this section perform remaining operations on
% the values but are done outside of the loop in order to take advantage of
% vectorization (that is, if verbosity is not enabled since if it was, then
% these operations must be computed immediately so that the correct
% stimulus location values can be plotted as early as possible).

% preallocate two columns for horizontal and vertical movements
stimulusLocationInEachFrame = NaN(numberOfFrames, 2);

for frameNumber = (1:numberOfFrames)
    
    frame = videoInput(:,:, frameNumber);
        
    correlation = normxcorr2(stimulus, frame);
        
    findPeakParametersStructure.enableGaussianFiltering = false;
    findPeakParametersStructure.stripHeight = frameHeight;
    [xPeak, yPeak, peakValue, ~] = ...
        FindPeak(correlation, findPeakParametersStructure);
        
    % Show surface plot for this correlation if verbosity enabled
    if parametersStructure.enableVerbosity
        figure(1);
        [surfX,surfY] = meshgrid(1:size(correlation,2), 1:size(correlation,1));
        surf(surfX, surfY, correlation,'linestyle','none');
        title([num2str(frameNumber) ' out of ' num2str(numberOfFrames)]);
        xlim([1 size(correlation,2)]);
        ylim([1 size(correlation,1)]);
        zlim([-1 1]);
        
        % Mark the identified peak on the plot with an arrow.
        text(xPeak, yPeak, peakValue, '\downarrow', 'Color', 'red', ...
            'FontSize', 20, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
        
        drawnow;
   
    end
    
    stimulusLocationInEachFrame(frameNumber,:) = [xPeak yPeak];
    
    % If verbosity is enabled, also show stimulus location plot with points
    % being plotted as they become available.
    if parametersStructure.enableVerbosity
        
        % Adjust for padding offsets added by normxcorr2()
        % If we enable verbosity and demand that we plot the points as we
        % go, then adjustments must be made here in order for the plot to
        % be interpretable.
        % Therefore, we will only perform these same operations after the
        % loop to take advantage of vectorization only if they are not
        % performed here, namely, if verbosity is not enabled and this
        % if statement does not execute.
        stimulusLocationInEachFrame(frameNumber,2) = ...
            stimulusLocationInEachFrame(frameNumber,2) - (frameHeight - 1);
        stimulusLocationInEachFrame(frameNumber,1) = ...
            stimulusLocationInEachFrame(frameNumber,1) - (frameWidth - 1);
        stimulusLocationInEachFrame(frameNumber,2) = ...
            stimulusLocationInEachFrame(frameNumber,2) - 1;
        stimulusLocationInEachFrame(frameNumber,1) = ...
            stimulusLocationInEachFrame(frameNumber,1) - 1;

        % Negate to flip directions.
        stimulusLocationInEachFrame(frameNumber,:) = -stimulusLocationInEachFrame(frameNumber,:);

        figure(2);
        plot(timeArray, stimulusLocationInEachFrame);
        title('Stimulus Locations');
        xlabel('Time (sec)');
        ylabel('Stimulus Locations (pixels)');
        legend('show');
        legend('Horizontal Location', 'Vertical Location');
    end
end

% Adjust for padding offsets added by normxcorr2()
% Do this after the loop to take advantage of vectorization
% Only run this section if verbosity was not enabled. If verbosity was
% enabled, then these operations were already performed for each point
% before it was plotted to the eye traces graph. If verbosity was not
% enabled, then we do it now in order to take advantage of vectorization.
if ~parametersStructure.enableVerbosity
    stimulusLocationInEachFrame(:,2) = ...
        stimulusLocationInEachFrame(:,2) - (frameHeight - 1);
    stimulusLocationInEachFrame(:,1) = ...
        stimulusLocationInEachFrame(:,1) - (frameWidth - 1);
    stimulusLocationInEachFrame(:,1) = stimulusLocationInEachFrame(:,1) - 1;
    stimulusLocationInEachFrame(:,2) = stimulusLocationInEachFrame(:,2) - 1;

    % Negate eye position traces to flip directions.
    stimulusLocationInEachFrame = -stimulusLocationInEachFrame;
end

%% Find mean and standard deviation of each frame

% Preallocate
meanOfEachFrame = zeros(numberOfFrames, 1);
standardDeviationOfEachFrame = zeros(numberOfFrames, 1);

for i = 1:numberOfFrames
    meanOfEachFrame(i) = mean2(videoInputArray(:,:,i));
    standardDeviationOfEachFrame(i) = std2(videoInputArray(:,:,i));
end

%% Save stimulus size
stimulusSize = size(stimulus);

%% Save to output mat file
save(outputFileName, 'stimulusLocationInEachFrame', 'stimulusSize', ...
    'meanOfEachFrame', 'standardDeviationOfEachFrame');

end

