function InitGUIHelper(mainHandles, GUIhandle)


% set all text objects to proper font sizes..
children = GUIhandle.Children.findobj('Type','uicontrol');
for i=1:length(children)
    if ~isempty(strfind(lower(children(i).Style),'text')) || ...
       ~isempty(strfind(lower(children(i).Style),'button')) || ...
       ~isempty(strfind(lower(children(i).Style),'checkbox'))
        children(i).FontUnits = 'points';
        children(i).FontSize = mainHandles.uiFontSize;
        children(i).FontWeight = 'normal';
    end
end


panelAndButtonGroups = GUIhandle.Children.findobj('Type','uipanel');
panelAndButtonGroups = [panelAndButtonGroups;...
    GUIhandle.Children.findobj('Type','uibuttongroup')];
for i=1:length(panelAndButtonGroups)
    panelAndButtonGroups(i).FontUnits = 'points';
    panelAndButtonGroups(i).FontSize = mainHandles.uiTitleFontSize;
    panelAndButtonGroups(i).FontWeight = 'bold';
end