function result = IsPositiveRealNumber(value)
%IS POSITIVE REAL NUMBER  Checks if value is a positive, real number.
%   Checks if value is a positive, real number. Returns true if so, else false.
%
%  MNA 11/300/19 made it array-compatible

result = IsRealNumber(value) & (value > 0);

