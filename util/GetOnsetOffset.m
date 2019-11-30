function [onsets, offsets] = GetOnsetOffset(indices)
%[onsets, offsets] = GetOnsetOffset(indices)
%
%

% find onset and offsets
if ~isrow(indices)
    indices = indices';
end

onsetOffset = [0 diff(indices)];
onsets = find(onsetOffset == 1);
offsets = find(onsetOffset == -1);

% address missing onset at the beginning
if offsets(1) < onsets(1)
    onsets = [1 onsets];
end

% address missing offset at the end
if offsets(end) < onsets(end)
    offsets = [offsets onsets(end)];
end
