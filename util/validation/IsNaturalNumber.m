function result = IsNaturalNumber(value)
%IS NATURAL NUMBER  Checks if value is a natural number.
%   Checks if value is a natural number. Returns true if so, else false.
%
%  MNA 11/30/19 made it array-compatible

result = ~isnan(value) & (value >= 0) & (rem(value,1)==0);

    
