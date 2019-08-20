To run the Oriented FAST and Rotated BRIEF (ORB) example, follow these steps:

1. Change your current working folder to example/ORB where source files
detectORBFeaturesOCV.cpp and extractORBFeaturesOCV.cpp are located

2. Create MEX-file for the detector from the source file:
>> mexOpenCV detectORBFeaturesOCV.cpp

3. Create MEX-file for the extractor from source file:
>> mexOpenCV extractORBFeaturesOCV.cpp

4. Run the test script:
>> testORBFeaturesOCV.m 
The test script uses the generated MEX-files.