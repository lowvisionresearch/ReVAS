function output = WhereToCompute(input, isGPU, isSingle)

% if isSingle 
%     input = single(input)/255;
% end

if isGPU
    output = gpuArray(input);
else
    output = input;
end