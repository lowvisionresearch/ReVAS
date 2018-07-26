function result = IsNaturalNumber(value)
%IS NATURAL NUMBER  Checks if value is a natural number.
%   Checks if value is a natural number. Returns true if so, else false.

if isnan(value) || ...
        value <= 0 || ...
        rem(value,1) ~= 0
    result = false;
else
    result = true;
end
    
end