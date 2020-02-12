function result = IsRealNumber(value)
%IS REAL NUMBER  Checks if value is a real number.
%   Checks if value is a real number. Returns true if so, else false.
%
%  MNA 11/30/19 made it array-compatible

result = ~isnan(value);

