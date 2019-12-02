function result = IsPositiveInteger(value)
%IS POSITIVE INTEGER NUMBER  Checks if value is a positive integer number.
%   Returns true if so, else false.
%
%  MNA 11/30/19 made it array-compatible

result = ~isnan(value) & (value > 0) & (rem(value,1)==0);