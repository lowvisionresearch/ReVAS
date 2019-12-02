function result = IsNonNegativeRealNumber(value)
%IS NON NEGATIVE REAL NUMBER  Checks if value is a non-negative, real number.
%   Checks if value is a non-negative, real number. Returns true if so, else false.
%
%  MNA 11/30/19 made it array-compatible

result = IsRealNumber(value) & (value > 0);
