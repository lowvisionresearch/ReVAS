/*
//////////////////////////////////////////////////////////////////////////////
// Declaration for OpenCV ORB detector wrapper
//
// Copyright 2015-2017 The MathWorks, Inc.
//
//////////////////////////////////////////////////////////////////////////////
*/
#ifndef _DETECTORBCORE_API_
#define _DETECTORBCORE_API_

#include "vision_defines.h"

EXTERN_C LIBMWCVSTRT_API int32_T detectORB_detect(uint8_T *inImg,
	int32_T nRows, int32_T nCols, int32_T isRGB,
	void **outKeypoints);

EXTERN_C LIBMWCVSTRT_API void detectORB_assignOutput(void *ptrKeypoints,
                               real32_T * outLoc,real32_T * outMetric,
                               real32_T * outScale, real32_T * outOrientation,
                               int32_T * outMisc);
#endif
