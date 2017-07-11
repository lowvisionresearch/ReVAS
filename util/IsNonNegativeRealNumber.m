function result = IsNonNegativeRealNumber(value)
%IS NON NEGATIVE REAL NUMBER  Checks if value is a non-negative, real number.
%   Checks if value is a non-negative, real number. Returns true if so, else false.

if ~IsRealNumber(value) || value < 0
    result = false;
else
    result = true;
end
    
end