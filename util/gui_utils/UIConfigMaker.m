function [GUIposition, uiFontSize,...
        uiTitleFontSize,revasColors] = UIConfigMaker

% GUI sizes and positions
GUIposition.revas = [0.2327    0.1714     0.5393     0.6867];
GUIposition.bandFiltParameters = [0.4256    0.1895    0.2125    0.4410];
GUIposition.coarseParameters = [0.4375    0.1552    0.2125    0.4410];
GUIposition.trimParameters = [0.4256    0.2752    0.1976    0.4610];
GUIposition.stripParameters = [0.0738    0.2248    0.3839    0.6210];
GUIposition.stimParameters = [0.4214    0.1800    0.2214    0.5229];
GUIposition.sacParameters = [0.1000    0.2819    0.3601    0.5162];
GUIposition.filtParameters = [0.1030    0.1305    0.3554    0.6638];
GUIposition.fineParameters = [0.0946    0.2114    0.3661    0.6190];
GUIposition.rerefParameters = [0.6310    0.1514    0.2006    0.7790];
GUIposition.parallelization = [0.6173    0.4190    0.1982    0.2705];
GUIposition.gammaParameters = [0.4181    0.3143    0.2135    0.3486];


% font sizes
uiFontSize = 14;
uiTitleFontSize = 18;


% bright color theme
revasColors.background = [255 255 255]/255;
revasColors.boxBackground = [245 245 245]/255;
revasColors.text = [0 0 0]/255;
revasColors.activeButtonBackground = [0 122 255]/255;
revasColors.activeButtonText = [255 255 255]/255;
revasColors.pushButtonBackground = [142 142 147]/255;
revasColors.pushButtonText = [255 255 255]/255;
revasColors.passiveButtonBackground = [255 255 255]/255;
revasColors.passiveButtonText = [0 122 255]/255;
revasColors.activeBorder = [0 122 255]/255; % NOT used
revasColors.passiveBorder = [142 142 147]/255; % NOT used
revasColors.abortButtonBackground = [255 59 48]/255;
revasColors.abortButtonText = [255 255 255]/255;


% % dark color theme
% revasColors.background = [0 0 0]/255;
% revasColors.boxBackground = [10 10 10]/255;
% revasColors.text = [255 255 255]/255;
% revasColors.activeButtonBackground = [255 149 0]/255;
% revasColors.activeButtonText = [255 255 255]/255;
% revasColors.pushButtonBackground = [0 0 0]/255;
% revasColors.pushButtonText = [255 255 255]/255;
% revasColors.passiveButtonBackground = [0 0 0]/255;
% revasColors.passiveButtonText = [255 149 0]/255;
% revasColors.activeBorder = [0 122 255]/255; % NOT used
% revasColors.passiveBorder = [142 142 147]/255; % NOT used
% revasColors.abortButtonBackground = [255 59 48]/255;
% revasColors.abortButtonText = [255 255 255]/255;


% save them in a config file
p = which('ReVAS');
save([fileparts(p) filesep 'uiconfig.mat'],'GUIposition','uiFontSize',...
    'uiTitleFontSize','revasColors');