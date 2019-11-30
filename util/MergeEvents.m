function [st, en] = MergeEvents(st, en, minInterEventInterval)
% [st, en] = MergeEvents(st, en, minInterEventInterval)
%
%


% loop until there is no pair of consecutive events closer than
% *minInterEventInterval*
while true
    interEventInterval = (st(2:end)-en(1:end-1));
    toBeMerged = find(interEventInterval < minInterEventInterval);
    if isempty(toBeMerged)
        break;
    end
    
    st(toBeMerged+1) = [];
    en(toBeMerged) = [];

end
