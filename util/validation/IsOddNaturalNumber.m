function result = IsOddNaturalNumber(value)
%IS NATURAL NUMBER  Checks if value is an odd natural number.
%   Checks if value is an odd natural number. Returns true if so, else false.

if ~IsNaturalNumber(value) || mod(value, 2) == 0
    result = false;
else
    result = true;
end
    
end