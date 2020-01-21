function params = RemoveFields(params,fieldList)
% params = RemoveFields(params,fieldList)
%
% Remove fields from struct. Use try/catch for each field since some of the
% fields may not exist in params.
% 
% MNA 1/2020

for i=1:length(fieldList)
    try
        params = rmfield(params,fieldList{i});
    catch
    end
end