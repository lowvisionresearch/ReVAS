function argument = ...
    ValidateArgument(params,argumentStr,defaultValue,validationFun,callerStr)
% argument = ValidateArgument(params,argumentStr,defaultValue,validationFun,callerStr)
%
%    Helper function to validate input arguments in various functions.
%


if ~isfield(params, argumentStr)
    argument = defaultValue;
    RevasWarning([callerStr ' is using default parameter for ' argumentStr ': ' num2str(argument)], params);
else
    argument = params.(argumentStr);
    if ~validationFun(argument)
        error([callerStr ': ' argumentStr 'must satisfy ' func2str(validationFun)]);
    end
end
