function result = IsOddNaturalNumber(value)
%IS NATURAL NUMBER  Checks if value is an odd natural number.
%   Checks if value is an odd natural number. Returns true if so, else false.
%
%  MNA 11/300/19 made it array-compatible

result = IsNaturalNumber(value) & (mod(value, 2) == 1);
