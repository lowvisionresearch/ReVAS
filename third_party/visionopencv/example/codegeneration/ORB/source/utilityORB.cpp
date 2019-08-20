//////////////////////////////////////////////////////////////////////////////
// Utility functions for OpenCV ORB detector and extractor
//
// Copyright 2015 The MathWorks, Inc.
//  
//////////////////////////////////////////////////////////////////////////////

#include "utilityORB.hpp"

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// This routine transforms OpenCV KeyPoint to fields of MATLAB Struct
////////////////////////////////////////////////////////////////////////////////
void orbKeyPointsToStruct(const std::vector<cv::KeyPoint> & keypoints,
                            real32_T * location,  real32_T * metric,
                            real32_T * scale,  real32_T * orientation,
                            int32_T * misc)
{
    const real32_T piOver180 = (real32_T)(CV_PI/180.0f);
    const int num = (int)keypoints.size();
    for (int32_T i = 0; i < num; ++i)
    {
        cv::KeyPoint& kp = (cv::KeyPoint&)keypoints[i];
        location[i]     = kp.pt.x + 1; // Convert to 1-based indexing
        location[i+num] = kp.pt.y + 1;
        metric[i]       = kp.response;
        scale[i]        = kp.size;
        orientation[i]  = kp.angle < 0.0f ? 0.0f : kp.angle * piOver180;
        misc[i]         = kp.class_id;
    }
}

