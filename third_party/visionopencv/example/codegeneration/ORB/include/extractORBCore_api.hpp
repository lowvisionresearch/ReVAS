/*
//////////////////////////////////////////////////////////////////////////////
// Declaration for OpenCV ORB extractor wrapper
//
// Copyright 2015-2017 The MathWorks, Inc.
//
//////////////////////////////////////////////////////////////////////////////
*/
#ifndef _EXTRACTORBCORE_API_
#define _EXTRACTORBCORE_API_

#include "vision_defines.h"

EXTERN_C LIBMWCVSTRT_API
int32_T extractORB_compute(const uint8_T * img, const int32_T nRows, const int32_T nCols,
                             real32_T * location, real32_T * metric,
                             real32_T * scale, real32_T * orientation,
                             int32_T * misc, const int32_T numKeyPoints,
                             void ** features, void ** keypoints);

EXTERN_C LIBMWCVSTRT_API
void extractORB_assignOutput(void *ptrDescriptors, void *ptrKeyPoints,
                               real32_T * location, real32_T * metric,
                               real32_T * scale, real32_T * orientation,
                               int32_T * misc, uint8_T * features);
#endif
