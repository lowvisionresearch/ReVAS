function indices = GetIndicesFromOnsetOffset(st,en,n)
% indices = GetIndicesFromOnsetOffset(st,en,n)

if length(st) ~= length(en)
    error('GetIndicesFromOnsetOffset: st and en must have same dimensions.');
end

indices = false(1,n);
for i=1:length(st)
    indices(st(i):en(i)) = true;
end


