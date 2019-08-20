//////////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for OpenCV ORB feature detector. 
// ORB stands for Oriented FAST and Rotated BRIEF. It is basically a fusion of 
// FAST keypoint detector and BRIEF descriptor. 
//
// Copyright 2014-2016 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"

using namespace cv;

//////////////////////////////////////////////////////////////////////////////
// Check inputs
//////////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
    }
    
    if (!mxIsUint8(prhs[0]))
    {       
        mexErrMsgTxt("Input image must be uint8.");
    }
}

///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    
    checkInputs(nrhs, prhs);

    // inputs
    cv::Ptr<cv::Mat> img = ocvMxArrayToImage_uint8(prhs[0], true);

    // get pointer to the algorithm
    Ptr<ORB> orbDetector = cv::ORB::create();
    
    // check if error occurs
    if( orbDetector.empty() )
        CV_Error(CV_StsNotImplemented, "OpenCV was built without ORB support");
    
    // prepare space for returned keypoints
    std::vector<KeyPoint> keypoints; 
    
    // invoke the detector
    orbDetector->detect(*img, keypoints); 
        
    // populate the outputs
    plhs[0] = ocvKeyPointsToStruct(keypoints);  
}

