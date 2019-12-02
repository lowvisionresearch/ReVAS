function report = RunTesters

testingFolder = fileparts(mfilename('fullpath'));
testerFunctions = dir([testingFolder filesep 'Tester_*.m']);

nTests = length(testerFunctions);
moduleBeingTested = cell(nTests,1);
score = false(nTests,1);
for i=1:nTests
    
    [~,testerFunctionStr] = fileparts(testerFunctions(i).name);
    moduleBeingTested{i} = testerFunctionStr((strfind(testerFunctionStr,'_')+1):end);
    score(i) = eval(testerFunctionStr);
end

report = table(score,'VariableNames',{'Passed'},'RowNames',moduleBeingTested);