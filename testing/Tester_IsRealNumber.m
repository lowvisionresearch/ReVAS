function success = Tester_IsRealNumber

if IsRealNumber(pi) && ...
        all(IsRealNumber(rand(1,10))) && ...
        ~IsRealNumber(nan) && ...
        any(IsRealNumber([nan nan 3]))
    success = true;
else
    success = false;
end