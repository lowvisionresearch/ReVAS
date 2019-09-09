//////////////////////////////////////////////////////////////////////////
// This example creates a C++ MEX-file for the GPU version of OpenCV's
// matchTemplate. Here matchTemplate uses normalized cross correlation method
// to search for matches between an image patch and an input image.
//
// This example requires the Parallel Computing Toolbox and a CUDA capable GPU.
//
// This example is only supported on Linux and Windows.
//
// Copyright 2014-2016 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////


// Include utilities for converting mxGPUArray to cv::gpu::GpuMat
#include "opencvgpumex.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "gpu/mxGPUArray.h"

///////////////////////////////////////////////////////////////////////////
// Return true if gpuArray data is UINT8 
///////////////////////////////////////////////////////////////////////////
boolean_T isGPUDataUINT8(const mxArray * in)
{

    // Get the underlying data type the gpuArray
    const mxGPUArray * img = mxGPUCreateFromMxArray(in);

    boolean_T isUINT8 = mxGPUGetClassID(img) == mxUINT8_CLASS;

    // Clean up gpuArray header
    mxGPUDestroyGPUArray(img);

    return isUINT8;
}

///////////////////////////////////////////////////////////////////////////
// Check inputs
///////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{    
    const mwSize * tdims, * fdims;

    // Check number of inputs
    if (nrhs != 2)
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
    }

    // Inputs must be gpuArrays
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]))
    {
        mexErrMsgTxt("Both input images must be stored in gpuArrays."); 
    }
    const mxGPUArray * img1 = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray * img2 = mxGPUCreateFromMxArray(prhs[1]);

    // Check input dimensions
    tdims = mxGPUGetDimensions(img1);
    fdims = mxGPUGetDimensions(img2);

    mwSize ndimsImg1 = mxGPUGetNumberOfDimensions(img1);
    mwSize ndimsImg2 = mxGPUGetNumberOfDimensions(img2);

	// Clean-up headers. Data is not deleted.
	mxGPUDestroyGPUArray(img1);
	mxGPUDestroyGPUArray(img2);

    if (ndimsImg1 > 2)
    {
        mexErrMsgTxt("Incorrect number of dimensions. First input must be a matrix.");
    }

    if (ndimsImg2 > 2)
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

    if (!isGPUDataUINT8(prhs[0]) || !isGPUDataUINT8(prhs[1]))
    {       
        mexErrMsgTxt("Input image must be a gpuArray with uint8 data.");
    }
}

///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    // Ensure MATLAB's GPU support is available.
    mxInitGPU();

    // Check inputs to mex function
    checkInputs(nrhs, prhs);

    // Convert mxArray inputs into OpenCV types
    cv::Ptr<cv::cuda::GpuMat> templateImgCV = ocvMxGpuArrayToGpuMat_uint8(prhs[0]);
    cv::Ptr<cv::cuda::GpuMat> imgCV         = ocvMxGpuArrayToGpuMat_uint8(prhs[1]);
    
    // Pad input image
    cv::cuda::GpuMat imgCVPadded((int)imgCV->rows + 2*(templateImgCV->rows - 1), 
            (int)imgCV->cols + 2*(templateImgCV->cols - 1), 
            CV_32FC1);
    cv::cuda::copyMakeBorder(*imgCV, 
            imgCVPadded, 
            templateImgCV->rows - 1, 
            templateImgCV->rows - 1,
            templateImgCV->cols - 1, 
           templateImgCV->cols - 1, 
           cv::BORDER_CONSTANT, 0);

    // Allocate output matrix
    int outRows = imgCV->rows - templateImgCV->rows + 1;
    int outCols = imgCV->cols - templateImgCV->cols + 1;
    cv::cuda::GpuMat outCV((int)outRows, (int)outCols, CV_32FC1);

    // Run the OpenCV template matching routine
    // CV_TM_CCOEFF_NORMED for normalized
    // CV_TM_CCOEFF for unnormalized
    // See https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
    cv::Ptr<cv::cuda::TemplateMatching> templateMatcher = cv::cuda::createTemplateMatching(CV_8UC3, CV_TM_CCOEFF_NORMED);
    templateMatcher->match(imgCVPadded, *templateImgCV, outCV);
    
    // Put the data back into the output MATLAB gpuArray
    plhs[0] = ocvMxGpuArrayFromGpuMat_single(outCV);
}

