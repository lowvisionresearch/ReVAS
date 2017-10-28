function result = IsPositiveRealNumber(value)
%IS POSITIVE REAL NUMBER  Checks if value is a positive, real number.
%   Checks if value is a positive, real number. Returns true if so, else false.

if ~IsRealNumber(value) || value <= 0
    result = false;
else
    result = true;
end
    
end