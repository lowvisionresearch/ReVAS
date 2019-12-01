function success = Tester_MakeStimulusCross


try
    s = MakeStimulusCross;
    
    crossSize = 11;
    crossThickness = 1;
    pos = MakeStimulusCross(crossSize, crossThickness);

    polarity = 0;
    neg = MakeStimulusCross(crossSize, crossThickness, polarity);

    if all(all(neg | pos)) && sum(s(:)) == 21
        success = true;
    else
        success = false;
    end

catch
    success = false;
end
