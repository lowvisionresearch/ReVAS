%% Specifications for Important Functions
%

%% Strip Analysis Module
% 
% *Purpose* 
% Extract eye movements in units of pixels.
%
% *Method* 
% Cross-correlation of horizontal strips with a pre-defined
% reference frame. 
%
% *Input arguments*
%%
% # Full path to the video *or* video as a 3D (or 4D array depending on the
% existence of color channels).
% # Full path to the reference frame image *or* the reference frame itself
% as a 2D (3D array, if there are color channels).
% # A parameters structure specifying all necessary parameters for strip
% analysis. Must-have fields: *strip height* and *strip width* in units of
% pixels, output *sampling rate* (which will be used to compute number of
% strips per frame), a *subpixel* flag to enable/disable interpolation, a
% sub-structure where subpixel interpolation parameters (neighborhood size
% and subpixel depth) will be stored, an *adaptive search* flag to
% enable/disable confined/adaptive search for cross-correlation peak
% (depending on the success of this module, adaptive search might not be
% needed), an array indicating *bad frames* (i.e., frames where image
% quality is so bad to perform strip analysis, or frames where subject
% blinked. These frames will be included in the strip analysis to save
% computation time), minimum *peak ratio* (a measure of confidence for the
% estimated location of each strip, see below for more), a verbosity flag
% (which will be used to display progress in real-time), and a bunch of
% axes handles to be used to display progress if verbosity is enabled.
%
% *Output arguments*
%%
% # Raw eye position traces (horizontal and vertical) in units of pixels.
% # Useful eye position traces (horizontal and vertical) in units of
% pixels. The difference between the raw and useful positions is as
% follows. When the peak ratio in a cross-correlation map of a certain
% strip is below the threshold specificied by the user, the corresponding
% eye position samples will be replaced by NaN in the useful eye position
% traces but they will be kept as is in the raw eye position traces. This
% is an important feature for, let's say, the user entered an
% optimistically high peak-ratio threshold but the video turned out to be
% of bad quality and more-than-expected number of samples were thrown away
% as unuseful. The user in this case can use the output of this function to
% retrospectively change the peak-ratio threshold and re-create useful eye
% position traces by using the raw traces *and* the peak-ratio for each
% strip stored in a structure (see below).
% # Corresponding time array in seconds (always starts at 0).
% # A structure where all stats regarding the analysis are kept. These
% stats include peak ratio, peak value, and search window for each and
% every strip, and an array of error structures which keep information
% about the runtime errors (caught by |try|/ |catch| statements) at certain
% strips. (Sometimes, the image quality is not good enough to get a nice
% cross-correlation map where the peak is well-defined. If the peak ratio
% is below the threshold specificed by the parameters structure, this strip
% will be skipped and the corresponding eye position sample in time will be
% NaN.) This structure can be used later to retrospectively inspect the
% quality of the analysis, say, if the user did not enable verbosity while
% running this module but later on was interested in looking at how the
% analysis went.
%
%
%%
% *Notes*
%%
% * This function will use |normxcorr2| function available in MATLAB for
% computing the cross-correlation maps. Since this function uses either
% frequency domain or time domain computation depending on the size of the
% arrays, it offers a good adaptive compromise. 
% * |normxcorr2| does
% not automatically do the computations in GPU. To do so, the strip and the
% reference frame must be transferred to GPU memory by using |gpuArray|
% function, also available in MATLAB. When computations are done in GPU,
% the output variables must be transferred back to the PC memory by using
% |gather| function. In this module, the cross-correlation map need not be
% transferred but the peak locations and the peak value should be
% transferred to properly store them in the third output argument described
% above.
% * 2D interpolation option can be implemented as a separate function since
% it may be needed in other modules of ReVAS. The specifications for this
% helper function is very straightforward. Input arguments: 2D correlation
% map, pixel coordinates of the peak location (e.g., x0 and y0), and a
% parameters structure which contains at least two fields (an odd-numbered
% neigborhood size, typically 7, and subpixel depth, typically 50). Output
% arguments: interpolated pixel coordinates (e.g., x1 and y1) and an error
% structure caught by |try|/ |catch|. If there no error has occured, this
% argument will simply be an empty array. As the interpolation method, use
% |spline| option in |interp2|.




