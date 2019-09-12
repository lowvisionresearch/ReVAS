This example is only supported on 64-bit platforms and requires the Parallel
Computing Toolbox.

To run the Template Matching GPU example, follow these steps:

1. Change your current working folder to third_party/visionopencv/TemplateMatchingGPU where 
source file matchTemplateOCV_GPU.cpp is located

2. Create MEX-file from the source file:

On PC:
>> mexOpenCV matchTemplateOCV_GPU.cpp -lgpu -lmwocvgpumex -largeArrayDims

On Linux/Mac:
>> mexOpenCV matchTemplateOCV_GPU.cpp -lmwgpu -lmwocvgpumex -largeArrayDims


3. Run the test script:
>> testMatchTemplateGPU.m 
The test script uses the generated MEX-file.

