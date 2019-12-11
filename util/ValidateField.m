function params = ValidateField(params,default,validate,callerStr)
% params = ValidateField(params,default,validate,callerStr)
%
%    Helper function to validate input arguments in various functions.
%

% get all field names
fields = fieldnames(default);

% go over fields
for i=1:length(fields)

    if ~isfield(params, fields{i})
        params.(fields{i}) = default.(fields{i});
        RevasWarning([callerStr ' is using default parameter for ' fields{i} ': ' num2str(params.(fields{i}))], params);
    else
        if ~validate.(fields{i})(params.(fields{i}))
            error([callerStr ': ' fields{i} ' must satisfy ' func2str(validate.(fields{i}))]);
        end
    end

end