

Requirements:
- CUDA toolkit (must match version returned by matlab command gpuDevice)
- working compiler (check with mexcuda -setup)


mexcuda -lcufft cuda_scratch.cpp helper/convolutionFFT2D.cu


