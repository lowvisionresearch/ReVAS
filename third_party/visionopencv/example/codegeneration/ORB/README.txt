This example shows how to integrate OpenCV functions into MATLAB Coder 
generated C/C++ code. It uses MATLAB coder's ExternalDependency class to 
integrate the OpenCV functions. To validate the output of the generated code,   
this example first computes the output of the handwritten MEX files. Then it
uses MATLAB Coder to generate the MEX target. The MEX target is selected so
that the functions in the generated code can be called from MATLAB. By setting 
proper properties of code generation configuration object (coder.config), 
the genrated code can be deployed as a standalone application. 

This example is a modified version of the "Detection and Extraction of ORB 
Features" example (located in the example/ORB folder). It is modified to 
support code generation. The code generation strategy followed in this 
example mimics the pattern used in the "Introduction to Code Generation with 
Feature Matching and Registration" example included in the Computer Vision 
System Toolbox.

To run the example:

1. If you have not yet built the handwritten MEX-files for ORB detector and 
extractor, please follow the steps in the example/ORB/README.txt. To find 
this README file:
   >> cd(fullfile(fileparts(which('mexOpenCV.m')),'example','ORB'))
   >> edit README.txt

2. Add the location of the handwritten MEX-files you created to the MATLAB path.
   >> addpath(fullfile(fileparts(which('mexOpenCV.m')),'example','ORB'))

3. Change your current working folder to example/codegeneration/ORB where 
source files ORBFeaturesCodegen_kernel.m is located.
   >> cd(fullfile(fileparts(which('mexOpenCV.m')),'example','codegeneration','ORB'))

4. Generate code for the MEX target from ORBFeaturesCodegen_kernel.m using 
MATLAB Coder. The generated code integrates OpenCV ORB detector and ORB  
extractor functions via wrapper functions, located under the 'source' folder.
   >> imageTypeAndSize = coder.typeof(uint8(0), [1000 1000],[true true]);
   >> codegen ORBFeaturesCodegen_kernel.m -args {imageTypeAndSize,imageTypeAndSize};

5. Refresh the file system cache.
   >> rehash toolbox

6. Validate the output of the generated code against the handwritten MEX output.
   >> testORBFeaturesCodegen

