function isDifferent = CompareFieldsHelper(p1,p2)
% isDifferent = CompareFieldsHelper(p1,p2)
%
%   Compare the fields of two structures. Returns false if there is any
%   difference.
%
% 

fields = fieldnames(p1);

isDifferent = false;
for fld = 1:length(fields)
    var1 = p1.(fields{fld});
    var2 = p2.(fields{fld});

    % first check lengths.. easiest check
    if length(var1) ~= length(var2)
        isDifferent = true;
        return;
    end

    % if lengths are the same, do an element-wise check for
    % equality. this should work for char arrays too.
    if ~bsxfun(@isequal,var1,var2)
        isDifferent = true;
        return;
    end
end
