//////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for OpenCV routine matchTemplate. 
// Here matchTemplate uses normalized cross correlation method to search 
// for matches between an image patch and an input image.
//
// Copyright 2014 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"
#include <opencv2/core/base.hpp>

#define _DO_NOT_EXPORT
#if defined(_DO_NOT_EXPORT)
#define DllExport  
#else
#define DllExport __declspec(dllexport)
#endif

///////////////////////////////////////////////////////////////////////////
// Check inputs
///////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{    
	const mwSize * tdims, *fdims;
        
    // Check number of inputs
    if (nrhs != 2)
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
    }
    
    // Check input dimensions
    tdims = mxGetDimensions(prhs[0]);
    fdims = mxGetDimensions(prhs[1]);
    
    if (mxGetNumberOfDimensions(prhs[0])>2)
    {
        mexErrMsgTxt("Incorrect number of dimensions. First input must be a matrix.");
    }
    
    if (mxGetNumberOfDimensions(prhs[1])>2)
    {
        mexErrMsgTxt("Incorrect number of dimensions. Second input must be a matrix.");
    }
    
    if (tdims[0] > fdims[0])
    {
        mexErrMsgTxt("Template should be smaller than image.");
    }
    
    if (tdims[1] > fdims[1])
    {
        mexErrMsgTxt("Template should be smaller than image.");
    }    
    
	// Check image data type
    if (!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]))
    {
        mexErrMsgTxt("Template and image must be UINT8.");
    }
}

///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
	// Check inputs to mex function
    checkInputs(nrhs, prhs);
    
    // Convert mxArray inputs into OpenCV types
    cv::Ptr<cv::Mat> templateImgCV = ocvMxArrayToImage_uint8(prhs[0], true);
    cv::Ptr<cv::Mat> imgCV         = ocvMxArrayToImage_uint8(prhs[1], true);
    
    // Pad input image
    cv::Mat imgCVPadded((int)imgCV->rows + 2*(templateImgCV->rows - 1), 
            (int)imgCV->cols + 2*(templateImgCV->cols - 1), 
            CV_32FC1);
    cv::copyMakeBorder(*imgCV, 
            imgCVPadded, 
            templateImgCV->rows - 1, 
            templateImgCV->rows - 1,
            templateImgCV->cols - 1, 
           templateImgCV->cols - 1, 
           cv::BORDER_CONSTANT, 0);

    // Allocate output matrix
    int outRows = imgCV->rows - templateImgCV->rows + 1;
    int outCols = imgCV->cols - templateImgCV->cols + 1;
    cv::Mat outCV((int)outRows, (int)outCols, CV_32FC1);

    // Run the OpenCV template matching routine
    // CV_TM_CCOEFF_NORMED for normalized
    // CV_TM_CCOEFF for unnormalized
    // See https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
    cv::matchTemplate(imgCVPadded, *templateImgCV, outCV, CV_TM_CCOEFF_NORMED );

    // Put the data back into the output MATLAB array
    plhs[0] = ocvMxArrayFromImage_single(outCV);

}
