%% Retinal Video Analysis Suite (ReVAS)
%
% This folder is the SVN Repository for the ReVAS.

%%
% ReVAS will consist of the following components:
%
% # Graphical User Interface (GUI) for performing all processing without
% any need for additional programming.
% # A series of functions to perform specific operations, which are called
% by the GUI, can be executed via MATLAB command line, or used by other
% third-party scripts/functions for batch processing.

%%
% The most typical workflow will be as follows:
%
% * User runs the GUI by typing |ReVAS| in the MATLAB command line.
% * User browses local folders and selects a single or multiple .avi files
% which has retinal videos obtained with a scanning laser ophthalmoscope.
% * User opens up a dialog box which lists the available set of processing
% steps and chooses a subset or all of the operations. Users can also
% re-arrange the order of each processing step in they choose to do so.
% * After user confirms the desired set of operations and their order, the
% GUI is populated with the relevant set of parameters in editable forms.
% All available parameter profiles are listed in a listbox and the user
% selects the desired profile. If there is no user-edited parameter
% profile, default values will be populated, which can be found in the
% *|defaultParameters.mat|* file under the ReVAS repository. If user makes
% any changes to any of the parameters, the GUI prompts the user if he/she
% wants to save the modified parameter set as a new profile. If so, user
% enters a filename for the new parameter set. Upon saving the new profile,
% the newly added profile is also added to the profile list. The profile
% list will be a field |profileList| of a structure called |settings| saved
% in |settings.mat| file. This file will also have other information
% regarding the configuration of the ReVAS (see below).
% * After user confirms the parameters to be used, another dialog box pops
% up and asks user to select the level of verbosity for the progress
% monitor. Last used levels of progress monitor will be saved in the
% |progressFlags| field of the |settings| structure saved in
% |settings.mat|. When ReVAS is launched for the first time, it will use
% some default progress report levels. But new selections for the level of
% progress report will overwrite the default values in the aforementioned
% field of the |settings| structure.
% * After the levels of progress report are specified for all operations
% selected, another dialog box pops up inquiring about whether or not to
% use parallelized or GPU processing, if they are available. If the PC does
% not have a CUDA compatible graphics card, or does not have more than one
% core in its CPU, then this dialog box does not appear.
% * Now, everything is set to go. Processing begins. Note that once the
% process starts, there is no way to pause or stop the process for the
% current video. It is, however, possible to pause processing before
% proceeding to the next video in the list.
% * Once all operations are completed, extracted eye movements and stimulus
% locations are presented to the user. User may choose to perform
% post-processing operations of eye movements. These will be done in a
% separate project solely devoted to eye movement analysis.
%
%

%% 
% # Mehmet
% # Derek
% # Matt
