//////////////////////////////////////////////////////////////////////////////
// OpenCV ORB extractor wrapper
//
// Copyright 2015 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////////

#include "extractORBCore_api.hpp"
#include "utilityORB.hpp"

#include "opencv2/opencv.hpp"
#include "cgCommon.hpp" // for cArrayToMat
                        // This file is located at:
                        //    <matlabroot>\toolbox\vision\builtins\src\ocv\include

////////////////////////////////////////////////////////////////////////////////
// Create cv::KeyPoints from fields of MATLAB Struct
////////////////////////////////////////////////////////////////////////////////
void structToORBKeyPoints(const real32_T * location, const real32_T * metric,
                            const real32_T * scale, const real32_T * orientation,
                            const int32_T * misc, const int32_T numKeyPoints,
                            std::vector<cv::KeyPoint> & keypoints)
{
    keypoints.reserve(numKeyPoints);
    const int32_T octave = 0;
    for (int32_T i = 0; i < numKeyPoints; ++i)
    {
        keypoints.push_back(cv::KeyPoint(location[i] - 1.0f,
                                         location[i+numKeyPoints] - 1.0f,
                                         scale[i], orientation[i], metric[i],
                                         octave, misc[i]));
    }
}

////////////////////////////////////////////////////////////////////////////////
// Invoke ORB compute method to extract ORB Features
////////////////////////////////////////////////////////////////////////////////
int32_T extractORB_compute(const uint8_T * img, const int32_T nRows, const int32_T nCols,
                             real32_T * location, real32_T * metric,
                             real32_T * scale, real32_T * orientation, int32_T * misc,
                             const int32_T numKeyPoints,
                             void ** features, void ** keypoints)
{

    using namespace cv;

    const bool isRGB = false; // only grayscale images are supported for ORB

    Ptr<Mat> mat = new Mat;

    // cArrayToMat is defined in cgCommon.hpp that is part of the Computer
    // Vision System Toolbox
    cArrayToMat<uint8_T>(img, nRows, nCols, isRGB, *mat);

    // get pointer to the algorithm
    Ptr<ORB> orbDescriptor = cv::ORB::create();

    if(orbDescriptor.empty()) {
        CV_Error(CV_StsNotImplemented, "OpenCV was built without ORB support");
    }

    // create KeyPoint vector
    std::vector<KeyPoint> * keypointPtr = new std::vector<KeyPoint>();
    *keypoints = (void *)keypointPtr;

    // copy keypoint data
    structToORBKeyPoints(location, metric, scale, orientation, misc,
                           numKeyPoints,*keypointPtr);

    Mat * descriptors = new Mat();
    *features = (void *)descriptors;

    // invoke ORB compute method to extract ORB Features
    orbDescriptor->compute(*mat, *keypointPtr, *descriptors);

    return static_cast<int32_T>(keypointPtr->size());
}

////////////////////////////////////////////////////////////////////////////////
// Copy data
////////////////////////////////////////////////////////////////////////////////
void extractORB_assignOutput(void *ptrDescriptors, void *ptrKeyPoints,
                               real32_T * location, real32_T * metric,
                               real32_T * scale, real32_T * orientation,
                               int32_T * misc, uint8_T * features)
{

    // copy feature data
    const cv::Mat & descriptors = *((cv::Mat *)ptrDescriptors);
    cArrayFromMat<uint8_T>(features, descriptors);

    // copy key point data
    const std::vector<cv::KeyPoint> & keypoints = *((std::vector<cv::KeyPoint> *)ptrKeyPoints);
    orbKeyPointsToStruct(keypoints, location, metric, scale, orientation, misc);

    // free memory
    delete((std::vector<cv::KeyPoint> *)ptrKeyPoints);
    delete((cv::Mat *)ptrDescriptors);

}
