This example is only supported on 64-bit platforms and requires the Parallel
Computing Toolbox.

To run the Oriented FAST and Rotated BRIEF (ORB) GPU example, follow these steps:

1. Change your current working folder to example/ORB_GPU where the source file,
detectORBFeaturesOCV_GPU.cpp, is located. 

2. Create MEX-file for the detector from the source file:

On PC:
>> mexOpenCV detectORBFeaturesOCV_GPU.cpp -lgpu -lmwocvgpumex -largeArrayDims

On Linux/Mac:
>> mexOpenCV detectORBFeaturesOCV_GPU.cpp -lmwgpu -lmwocvgpumex -largeArrayDims

3. Run the test script:
>> testORBFeaturesOCV_GPU.m 
The test script uses the generated MEX-files.
