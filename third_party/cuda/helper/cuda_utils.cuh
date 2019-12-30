#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


// Helper functions for CUDA
#include "helper_functions.h"
#include "helper_cuda.h"
#include "convolutionFFT2D_common.h"

__global__ void normed_max(float *data, float *var, int fftH_x_fftW, 
                float *out_peak, float *out_secondpeak, int *out_argmax);
        
int snapTransformSize(int dataSize);

__global__ void sum(float* input, float* output);
//
void sum_call(float* input, float* output, int size);

//__global__ void sumCommSingleBlock(float* input, float* out, int arraySize, int blockSize);
//
//void sum_single_call(float* a, float* out, int arraySize, int blockSize);

void reduce6_call(float* g_idata, float* g_odata, unsigned int n);

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T* g_idata, T* g_odata, unsigned int n);

//void compute_sum(float* d_data_in, float* d_data_out);

void sum_16_call(float* in, float* out);

__global__ void sum_16(float* in, float* out);



void var(float* d_Data, float* d_sum_data, float* var_output, int kernelW, int kernelH, int dataW, int dataH);

__global__ void var_helper(float* in, float* store, float* sum, int size);

void normalized_call(float* corr, float* var, float* mean, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH);

//__global__ void normalized(float* corr, float* sum, float* var, float* mean, int size, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH, int current_line);
__global__ void normalized(float* corr, float* var, float* mean, int size, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH, int current_line);

void sum_less_threads(float* in, float* out, int size);

void sum_less_threads_mean(float* in, float* out, int size, int divide_num);

__global__ void divide_size(float* in, float* out, float size);

void reduce6_max_call(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, unsigned int n);

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_max(T* g_idata, T* g_odata, int* x_loc, unsigned int n);

void reduce3_max_call(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, unsigned int n);

void cuda_max(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, int h, int w);


template <class T>
__global__ void
reduce3_max(T* g_idata, T* g_odata, int* x_loc, unsigned int n);


__global__ void test_yz_call(float* data, float* val, int* loc, int size);
void yz_max(float* g_idata, float* final_data, int* final_index, float* buffer_data, int* buffer_index, int fftH, int fftW);
