This version is adapted from the Template Matching example that ships with
the support package.

It performs padding prior to correlating, in order to copy the behavior
of normxcorr2().

To run, follow these steps:

1. Change your current working folder to third_party/visionopencv/TemplateMatching where 
source file matchTemplateOCV.cpp is located

2. Create MEX-file from the source file:
>> mexOpenCV matchTemplateOCV.cpp

3. Run the test script:
>> testMatchTemplate.m 
The test script uses the generated MEX-file.