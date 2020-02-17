function isDifferent = CompareFieldsHelper(p1,p2)
% isDifferent = CompareFieldsHelper(p1,p2)
%
%   Compare two structures. Returns false if there is any difference.
%
% 

fields1 = fieldnames(p1);
fields2 = fieldnames(p2);

if length(fields1) ~= length(fields2)
    isDifferent = true;
    return;
end

isDifferent = false;
for fld = 1:length(fields1)
    var1 = p1.(fields1{fld});
    var2 = p2.(fields1{fld});

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
