function UIConfigMaker

revasOuterPosition = [ 0.2327 0.1714 0.5393 0.6867];
uiFontSize = 14;
uiTitleFontSize = 18;
revasColors.background = [255 255 255]/255;
revasColors.boxBackground = [255 255 255]/255;
revasColors.text = [0 0 0]/255;
revasColors.activeButtonBackground = [0 122 255]/255;
revasColors.activeButtonText = [255 255 255]/255;
revasColors.pushButtonBackground = [142 142 147]/255;
revasColors.pushButtonText = [255 255 255]/255;
revasColors.passiveButtonBackground = [255 255 255]/255;
revasColors.passiveButtonText = [0 122 255]/255;
revasColors.activeBorder = [0 122 255]/255;
revasColors.passiveBorder = [142 142 147]/255;
revasColors.abortButtonBackground = [255 59 48]/255;
revasColors.abortButtonText = [255 255 255]/255;

save([pwd '/gui/uiconfig.mat'],'revasOuterPosition','uiFontSize',...
    'uiTitleFontSize','revasColors');