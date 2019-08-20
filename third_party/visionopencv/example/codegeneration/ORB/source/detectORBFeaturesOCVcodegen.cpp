//////////////////////////////////////////////////////////////////////////////
// OpenCV ORB detector wrapper
//
// Copyright 2015-2017 The MathWorks, Inc.
//
//////////////////////////////////////////////////////////////////////////////

#include "detectORBCore_api.hpp"
#include "utilityORB.hpp"

#include "opencv2/opencv.hpp"
#include "cgCommon.hpp" // for cArrayToMat
                        // This file is located at:
                        //    <matlabroot>\toolbox\vision\builtins\src\ocv\include
using namespace cv;

//////////////////////////////////////////////////////////////////////////////
// Invoke OpenCV ORB detect method
//////////////////////////////////////////////////////////////////////////////
int32_T detectORB_detect(uint8_T *inImg,
    int32_T nRows, int32_T nCols, int32_T isRGB,
    void **outKeypoints)
{
    // use OpenCV smart pointer to manage the image data
    cv::Ptr<cv::Mat> inImage = new cv::Mat;
    bool isRGB_ = (bool)(isRGB != 0);

    // cArrayToMat is defined in cgCommon.hpp that is part of the Computer
    // Vision System Toolbox
    cArrayToMat<uint8_T>(inImg, nRows, nCols, isRGB_, *inImage);

    // keypoints
    std::vector<KeyPoint> *ptrKeypoints = (std::vector<KeyPoint> *)new std::vector<KeyPoint>();
    *outKeypoints = ptrKeypoints;
    std::vector<KeyPoint> &refKeypoints = *ptrKeypoints;

    // get pointer to the algorithm
    Ptr<ORB> orbDetector = cv::ORB::create();

    if(orbDetector.empty()) {
        CV_Error(CV_StsNotImplemented, "OpenCV was built without ORB support");
    }

    // invoke the detector
    orbDetector->detect(*inImage, refKeypoints);

    return ((int32_T)(refKeypoints.size()));
}

void detectORB_assignOutput(void *ptrKeypoints,
    real32_T *outLoc, real32_T *outMetric,
    real32_T * outScale,  real32_T * outOrientation,
    int32_T * outMisc)
{
    vector<KeyPoint> keypoints = ((std::vector<KeyPoint> *)ptrKeypoints)[0];

    // Populate the outputs
    orbKeyPointsToStruct(keypoints, outLoc, outMetric, outScale,
            outOrientation, outMisc);

    // free memory
    delete((std::vector<KeyPoint> *)ptrKeypoints);
}
