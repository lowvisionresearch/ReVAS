function output = WhereToCompute(input, isGPU)

if isGPU
    output = gpuArray(input);
else
    output = input;
end