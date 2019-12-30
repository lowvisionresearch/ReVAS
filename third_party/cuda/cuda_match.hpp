#ifndef WIZARD_OPTIMIZED_CUDA_CROSS_CORRELATION_TECHNIQUE_HPP
#define WIZARD_OPTIMIZED_CUDA_CROSS_CORRELATION_TECHNIQUE_HPP

#include "helper/I_Template_Matching_Technique.hpp"

#include <memory>

//cuda include
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "helper/helper_functions.h"
#include "helper/helper_cuda.h"
#include "helper/cuda_utils.cuh"
#include "helper/convolutionFFT2D_common.h"

//! A "precomputed" version of the above. Useful when registering multiple patterns against the same search space.
class Yz_Helper {
public:
    //! Constructor, given the search space we want to look through, and the size of the patterns that will be provided later.
    Yz_Helper(const cv::Mat& search_space, cv::Size pattern_size);

/*    //! Perform a match against the `search_space` given earlier.
    void match(const cv::Mat& pattern, cv::Point& output_loc, double& output_peak_size) const;

	void match_time_stat(const cv::Mat& pattern, cv::Point& output_loc, double& output_peak_size, double *datapreptime, double *calculationtime, double *maxfindingtime) const;

private:
    //! The "padded" image that we actually cross correlate against
    cv::Mat m_padded_img;

    //! How much padding was applied on each of the x directions (left and right sides)
    double m_padding_x;

    //! How much padding was applied on each of the y directions (top and bottom sides)
    double m_padding_y;
    float
        * h_Data,
        *h_Kernel,
        *h_ResultGPU,
        *test_sum,
        *test_var;


    float
        * d_Data,
        *d_PaddedData,
        *d_Kernel,
        *d_PaddedKernel,
        *d_test_sum,
        *d_test_var,
        *d_template_mean_buffer,
        *d_template_mean;

    fComplex
        * d_DataSpectrum,
        *d_KernelSpectrum;

    cufftHandle
        fftPlanFwd,
        fftPlanInv;

    int kernelH;
    int kernelW;
    int kernelY;
    int kernelX;
    int   dataH;
    int   dataW;
    int    fftH;
    int    fftW;

	//for cuda max
	float
		* h_data,
		* h_max,
		* h_max1;

	int
		* h_x_loc,
		* h_y_loc;

	float
		* d_data,
		* d_max,
		* d_max1;

	int
		* d_x_loc,
		* d_y_loc;

	int H;
	int W;
	int size; */
};


#endif // WIZARD_OPTIMIZED_CUDA_CROSS_CORRELATION_TECHNIQUE_HPP