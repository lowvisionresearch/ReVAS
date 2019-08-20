//////////////////////////////////////////////////////////////////////////////
// Declaration for utility function
//
// Copyright 2015-2017 The MathWorks, Inc.
//
//////////////////////////////////////////////////////////////////////////////
#ifndef _UTILITYORB_
#define _UTILITYORB_

#include "opencv2/opencv.hpp"
#include "vision_defines.h"

void orbKeyPointsToStruct(const std::vector<cv::KeyPoint> & keypoints,
                            real32_T * location,  real32_T * metric,
                            real32_T * scale,  real32_T * orientation,
                            int32_T * misc);
#endif
