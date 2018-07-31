function BrowseRefs
% BrowseRefs
%  This function can be used to browse coarserefs from already analyzed
%  videos. It asks user to select folder(s) and then displays each
%  coarseref file in each folder one at a time. If the user clicks
%  somewhere inside the image, that file is set aside to be used later on.
%  If the user clicks somewhere outside the image, that file is skipped.
%
%  After selecting a good collection of ref. frames, the user can use
%  MakeBigMontage.m function to combine them to make a better global
%  reference frame.
%
%
%  MNA 4/18/17 wrote the initial version
%  MNA 7/18/18 put some comments


if nargin<1
    foldername = uipickfiles;
    if ~iscell(foldername)
        if foldername == 0 
            fprintf('User cancelled folder selection. Silently exiting...\n');
            return;
        end
        foldername = {foldername};
    end
end

parentPath = fileparts(foldername{1});
fileCounter = 0;

% go over each folder if there are more than one folder selected
for i=1:length(foldername)

    currentFolder = foldername{i};
    
    % see which files exist in the current folder
    listing = dir(currentFolder);
    
    % get indices of all files (but not folders)
    indices = find(~[listing.isdir]); 
    
        % loop over all files in the selected folder
        for j=1:length(indices)

            % get next file name
            filename = [currentFolder filesep listing(indices(j)).name];
            
            % use the file only if does it have "coarseref" phrase in its name
            keyPhrase = strfind(filename,'coarseref');
            if ~isempty(keyPhrase)
                try
                    % load the local reference frame
                    load(filename,'referenceimage','referencematrix');
                    
                    figure(45678);
                    imshow(uint8(referenceimage));
                    [m,n]=size(referenceimage);
                    title('Click anywhere on the image to accept it. Otherwise click elsewhere.')
                    try
                        [x,y]=ginput(1);
                    catch
                        % figure closed
                        fprintf('Figure closed by user. silently exiting..\n');
                        return;
                    end
                    
                    if x>0 && y>0 && x<=m && y<=n
                        fileCounter = fileCounter+1;
                        save(sprintf('%s\\%d_coarseref.mat',parentPath,fileCounter),'referenceimage','referencematrix','filename');
                        fprintf('File saved\n');
                    else
                        fprintf('File ignored.\n');
                    end
                    
                catch 
                    fprintf('Local reference frame does not exist, skipping the file\n');
                end
            end
        end
end