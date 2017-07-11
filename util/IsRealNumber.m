function result = IsRealNumber(value)
%IS REAL NUMBER  Checks if value is a real number.
%   Checks if value is a real number. Returns true if so, else false.

if isnan(value)
    result = false;
else
    result = true;
end
    
end