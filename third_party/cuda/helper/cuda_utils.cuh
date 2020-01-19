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


// normalize correlation map and return argmax and max and 2nd max values
void normed_argmax(float *data, float *invvar,
                  float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes,
                  int fftH_x_fftW, float normnum, bool copyMap);
__global__ void norm_kernel(float *data, float *invvar,
                float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes,
                int fftH_x_fftW, float normnum, bool copyMap);
__global__ void argmax_kernel(float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes, int nblocks);


__global__ void test_yz_call(float* data, float* val, int* loc, int size);
template <class T>
__global__ void
reduce3_max(T* g_idata, T* g_odata, int* x_loc, unsigned int n);
void yz_max(float* g_idata, float* final_data, int* final_index, float* buffer_data, int* buffer_index, int fftH, int fftW);



//helper functions
int snapTransformSize(int dataSize);
void printmat(float * mat, int H, int W, char * name);
void printcmat(fComplex * mat, int H, int W, char * name);