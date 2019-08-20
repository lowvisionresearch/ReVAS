//////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for ORB feature descriptors extractor in OpenCV. 
// ORB stands for Oriented FAST and Rotated BRIEF. It is basically a fusion of 
// FAST keypoint detector and BRIEF descriptor. 
//
// Copyright 2014-2016 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"

using namespace cv;


//////////////////////////////////////////////////////////////////////////////
// Check inputs
//////////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
    }
    
    if (!mxIsUint8(prhs[0]))
    {       
        mexErrMsgTxt("Input image must be uint8.");
    }
}

//////////////////////////////////////////////////////////////////////////////
// The main MEX function entry point
//////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    checkInputs(nrhs, prhs);

    // inputs
    cv::Ptr<cv::Mat> img = ocvMxArrayToImage_uint8(prhs[0], true);

    std::vector<KeyPoint> keypoints;
    ocvStructToKeyPoints(prhs[1], keypoints);
    
    // get pointer to the algorithm
    Ptr<ORB> orbDescriptor = ORB::create();
    
    // check if error occurs
    if( orbDescriptor.empty() )
        CV_Error(CV_StsNotImplemented, "OpenCV was built without ORB support");
    
    // get descriptors
    Mat descriptors;    
    orbDescriptor->compute(*img, keypoints, descriptors); 

    // populate the outputs 
    plhs[0] = ocvMxArrayFromImage_uint8(descriptors);
    plhs[1] = ocvKeyPointsToStruct(keypoints);
}

