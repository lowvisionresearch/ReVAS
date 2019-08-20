//////////////////////////////////////////////////////////////////////////////
// This example creates C++ MEX-file for OpenCV ORB feature detector that runs
// on the GPU.  ORB stands for Oriented FAST and Rotated BRIEF. It is basically
// a fusion of FAST keypoint detector and BRIEF descriptor. 
//
// This example requires the Parallel Computing Toolbox and a CUDA capable GPU.
//
// This example is only supported on Linux and Windows.
//
// Copyright 2014-2016 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////////

// Include utilities for converting mxGPUArray to cv::cuda::GpuMat
#include "opencvgpumex.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"

#include "gpu/mxGPUArray.h"


using namespace cv;

//////////////////////////////////////////////////////////////////////////////
// Check inputs
//////////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 1 inputs.");
    }

    if (!mxIsGPUArray(prhs[0]))
    {
        mexErrMsgTxt("Input image must be a gpuArray.");
    }
    
    // Get the underlying data type of the gpuArray
    const mxGPUArray * img = mxGPUCreateFromMxArray(prhs[0]);
    boolean_T isUINT8 = mxGPUGetClassID(img) == mxUINT8_CLASS;
    mxGPUDestroyGPUArray(img);
    
    if (!isUINT8)
    {       
        mexErrMsgTxt("Input image must be a gpuArray with uint8 data.");
    }

}

///////////////////////////////////////////////////////////////////////////////
// Return an mxArray that contains the keypoint location data.
//
// The keypoint data is copied into a mxGPUArray and returned to MATLAB as a
// gpuArray. This leaves the data on the GPU for further processing in MATLAB.
//
// Other keypoint data (response, angle, etc) can be returned in a similar
// manner. 
///////////////////////////////////////////////////////////////////////////////
mxArray * getORBKeypointLocations(cv::cuda::GpuMat& keypoints)
{
    int numKeypoints = keypoints.cols;

    // Create the 2D mxGPUArray for the output locations
    mwSize dims[] = {static_cast<mwSize>(numKeypoints), 2};

    mxGPUArray * mxGPULocations = 
        mxGPUCreateGPUArray(2, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    float * const locations = 
        (float *)mxGPUGetData(mxGPULocations);

    // Create a GpuMat header around the mxGPUArray data. 
    cv::cuda::GpuMat outputLocations(2, numKeypoints, CV_32FC1, locations);

    // Create a GpuMat header to reference the keypoint location information,
    // which make up the first 2 rows of the keypoints matrix. 
    cv::Range rowRange(cv::cuda::ORB::X_ROW,
            cv::cuda::ORB::RESPONSE_ROW);
    
    cv::Range colRange = cv::Range::all();

    cv::cuda::GpuMat gpuLocations(keypoints, rowRange, colRange); 
    
    // Add 1 to the location for 1-based indexing.
    cv::cuda::add(gpuLocations, 1.0F, outputLocations);
   
    // Move the mxGPUArray to a mxArray
    mxArray * mxOutput = mxGPUCreateMxArrayOnGPU(mxGPULocations);

    // Cleanup the mxGPUArray header. The GPU data is not destroyed.
    mxGPUDestroyGPUArray(mxGPULocations);

    return mxOutput;
}

///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{  
    
    // Ensure MATLAB's GPU support is available.
    mxInitGPU();

    checkInputs(nrhs, prhs);

    // Convert gpuArray into cv::gpu::GpuMat 
    Ptr<cv::cuda::GpuMat> gpuImage = ocvMxGpuArrayToGpuMat_uint8(prhs[0]);
        
    // Set up the detector (using default algorithm parameters)
    static cv::Ptr<cv::cuda::ORB> featureDetector = cv::cuda::ORB::create();
    
    cv::cuda::GpuMat keypoints;  // detector output

    // Run the ORB detector on the GPU
    featureDetector->detectAsync(*gpuImage, keypoints, cv::cuda::GpuMat());

    // Return data as gpuArray types. This keeps keypoint data on the GPU for
    // further processing in MATLAB. 
    plhs[0] = getORBKeypointLocations(keypoints);
}



