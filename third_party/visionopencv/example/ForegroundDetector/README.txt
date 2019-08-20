To run the foreground example, follow these steps:

1. Change your current working folder to example/ForegroundDetector where 
source file backgroundSubtractorOCV.cpp is located

2. Create MEX-file from the source file:
>> mexOpenCV backgroundSubtractorOCV.cpp

3. Run the test script:
>> testBackgroundSubtractor.m 
The test script uses the class backgroundSubtractor which, in turn, uses the 
generated MEX-file.
