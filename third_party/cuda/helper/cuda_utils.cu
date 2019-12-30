#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper_cuda.h"

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "cuda_utils.cuh"
				
/*      
// constants for normed_max
// TODO: optimize
//cudaGetDeviceProperties(&prop, 0);
#define DSIZE 4096*4096 // maximum data size
#define nTPB 1024 // must be power of 2
#define MAX_KERNEL_BLOCKS 30 
#define MAX_BLOCKS ((DSIZE/nTPB)+1)
#define MIN(a,b) ((a>b)?b:a)
__device__ volatile float blk_peaks[MAX_BLOCKS];
__device__ volatile float blk_secondpeaks[MAX_BLOCKS];
__device__ volatile int   blk_argmaxes[MAX_BLOCKS];
__device__ int   blk_num = 0;

        
// see https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat
// TODO: only go over relevant portion of data
__global__ void normed_max(float *data, float *vars, int fftH_x_fftW, float *out_peak, float *out_secondpeak, int *out_argmax){

    return;
    
 /* __shared__ volatile float peaks[nTPB];
  __shared__ volatile float secondpeaks[nTPB];
  __shared__ volatile int argmaxes[nTPB];
  __shared__ volatile int last_block;
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  last_block = 0;
  float peak = 0;
  float secondpeak = 0;
  int argmax = -1;
  // sweep from global memory
  while (idx < fftH_x_fftW){  
    if (data[idx]/vars[idx] > peak) {
        secondpeak = peak; 
        peak = data[idx];
        argmax = idx;
    }
    idx += blockDim.x*gridDim.x;}
  // populate shared memory
  peaks[threadIdx.x] = peak;
  secondpeaks[threadIdx.x] = secondpeak;
  argmaxes[threadIdx.x] = argmax;
  __syncthreads();
  // sweep in shared memory
  for (int i = (nTPB>>1); i > 0; i>>=1){
    if (threadIdx.x < i)
      if (peaks[threadIdx.x] < peaks[threadIdx.x + i]){
          secondpeaks[threadIdx.x] = peaks[threadIdx.x] > secondpeaks[threadIdx.x+i]? 
                                     peaks[threadIdx.x] : secondpeaks[threadIdx.x+i]; 
          peaks[threadIdx.x] = peaks[threadIdx.x+i]; 
          argmaxes[threadIdx.x] = argmaxes[threadIdx.x+i];
      }
    __syncthreads();
  }
  // perform block-level reduction
  if (!threadIdx.x){
    blk_peaks[blockIdx.x] = peaks[0];
    blk_secondpeaks[blockIdx.x] = secondpeaks[0];
    blk_argmaxes[blockIdx.x] = argmaxes[0];
    if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
      last_block = 1;}
  __syncthreads();
  if (last_block){
    idx = threadIdx.x;
    peak = 0;
    secondpeak = 0;
    argmax = -1;
    while (idx < gridDim.x){
      if (blk_peaks[idx] > peak) {
          secondpeak = peak > blk_secondpeaks[idx]?
                       peak : blk_secondpeaks[idx];
          peak = blk_peaks[idx];
          argmax = blk_argmaxes[idx]; 
      }
      idx += blockDim.x;}
  // populate shared memory
    peaks[threadIdx.x] = peak;
    secondpeaks[threadIdx.x] = secondpeak;
    argmaxes[threadIdx.x] = argmax;
    __syncthreads();
  // sweep in shared memory
    for (int i = (nTPB>>1); i > 0; i>>=1){
      if (threadIdx.x < i)
        if (peaks[threadIdx.x] < peaks[threadIdx.x + i]) {
            secondpeaks[threadIdx.x] = peaks[threadIdx.x] > secondpeaks[threadIdx.x+i]?
                                       peaks[threadIdx.x] : secondpeaks[threadIdx.x+i];
            peaks[threadIdx.x] = peaks[threadIdx.x+i];
            argmaxes[threadIdx.x] = argmaxes[threadIdx.x+i];
        }
      __syncthreads();}
    if (!threadIdx.x)
       *out_peak = peaks[0];
       *out_secondpeak = secondpeaks[0];
       *out_argmax = argmaxes[0];
    }
}     */   
        
int snapTransformSize(int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;
    dataSize = iAlignUp(dataSize, 16);
    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
            {break;}
    lowPOT = 1U << hiBit;
    if (lowPOT == (unsigned int)dataSize)
        {return dataSize;}
    hiPOT = 1U << (hiBit + 1);
    return (hiPOT <= 1024)? hiPOT : iAlignUp(dataSize, 512);
}
        
void sum_call(float* input, float* output, int size) {
	sum << <1, size / 2 >> > (input, output);

}

__global__ void sum(float* input, float* output)
{
	const int tid = threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	//printf("%f",tid);
	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}
		__syncthreads();
		//modify fo purpose for odd number of threads
		if (number_of_threads % 2 == 1) {
			input[(number_of_threads - 2) * step_size * 2] += input[(number_of_threads - 1) * step_size * 2];
		}

		step_size <<= 1;
		number_of_threads >>= 1;
	}
	output[0] = input[0];
}


//copy from reduction sample
#include <stdio.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T* ()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T* () const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double* ()
	{
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}

	__device__ inline operator const double* () const
	{
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}
};


void reduce6_call(float* g_idata, float* g_odata, unsigned int n) {
	int threads = 1024;
	int blocks = n / 2048 + 1 > 64 ? 64 : n / 512 + 1;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6<float, 1024, true> << < dimGrid, dimBlock, smemSize >> > (g_idata, g_odata, n);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T* g_idata, T* g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			mySum += tile32.shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_square(T* g_idata, T* g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i] * g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i + blockSize] * g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
		{
			mySum += tile32.shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


void normalized_call(float* corr, float* var, float* mean, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH) {
	int threads = 1024;
	int blocks = 1024;
	for (int i = 0; i < fftH; i += 512) {
		normalized << <blocks, threads >> > (corr + i * fftW, var + i * (dataW - kernelX), mean, dataW - kernelX, kernelW, kernelH, dataW, dataH, kernelX, kernelY, fftW, fftH, i);
	}
}

/*
void normalized_call(float* corr, float* sum, float* var, float* mean, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH) {
	int threads = 1024;
	int blocks = 1024;
	//int threads = fftW;
    //int blocks = fftH;
    for (int i = 0; i < fftH; i += 512) {
		//normalized << <blocks, threads >> > (corr + i * fftW, sum + i * (dataW - kernelX), var + i * (dataW - kernelX), mean, dataW - kernelX, kernelW, kernelH, dataW, dataH, kernelX, kernelY, fftW, fftH, i);
        normalized << <blocks, threads >> > (corr + i * fftH, var + i * (dataH - kernelY), mean, dataH - kernelY, kernelH, kernelW, dataH, dataW, kernelY, kernelX, fftH, fftW, i);
	}
}*/

__global__ void normalized(float* corr, float* var, float* mean, int size, int kernelW, int kernelH, int dataW, int dataH, int kernelX, int kernelY, int fftW, int fftH, int current_line) {
	int index = threadIdx.x + blockIdx.x * 1024;
	int index_y = index / 2048;
	int index_x = index - 2048 * index_y;
	if (current_line + index_y >= (dataH - kernelY)) {
		if (current_line + index_y < fftH) {
			corr[index_y * fftW + index_x] = -1000000000.0; //get rid of region we don't want
		}
		return;
	}
	if (index_x < size) {
		//corr[index] = (corr[index] - mean * sum[index]) / var[index];
		corr[index_y * fftW + index_x] = corr[index_y * fftW + index_x] / var[index_y * (dataW - kernelX) + index_x];
		//in case of zero window variance
		if (var[index_y * (dataW - kernelX) + index_x] <= powf(10, -10)) {
			corr[index_y * fftW + index_x] = 0;
		}
	}
	else {
		corr[index_y * fftW + index_x] = -1000000000.0; //get rid of region we don't want
	}


}
/*
void normalized_custompadding_call(float* corr, float* var, int kernelW, int kernelH, int dataW, int dataH, int fftW, int fftH) {
	int threads = 1024;
	int totalsize = fftW * fftH;
	int blocks = totalsize / threads;
	normalized_custompadding << <blocks, threads >> > (corr, var, kernelW, kernelH, dataW, dataH, fftW, fftH);
}

__global__ void normalized_custompadding(float* corr, float* var, int kernelW, int kernelH, int dataW, int dataH, int fftW, int fftH) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = index / fftW;
	int index_x = index % fftW;
	if (index_y >= (dataH + kernelH - 1)) {
		if (index_y < fftH) {
			corr[index_y * fftW + index_x] = 0.0; //get rid of region we don't want
		}
		return;
	}
	if (index_x < (dataW + kernelW - 1)) {
		//corr[index] = (corr[index] - mean * sum[index]) / var[index];
		corr[index_y * fftW + index_x] = corr[index_y * fftW + index_x] / var[index_y * (dataW + kernelW - 1) + index_x];
		//in case of zero window variance
		if (var[index_y * (dataW + kernelW - 1) + index_x] <= powf(10, -8)) {
			corr[index_y * fftW + index_x] = 0;
		}
	}
	else {
		corr[index_y * fftW + index_x] = 0.0; //get rid of region we don't want
	}
}
*/
/*
void normalized_custompadding_sigma_call(float* corr, float* var, int kernelW, int kernelH, int dataW, int dataH, int fftW, int fftH, float* result_mean, float* result_sd, float* buffer) {
	int threads = 1024;
	int totalsize = fftW * fftH;
	int blocks = totalsize / threads;
	normalized_custompadding_sigma << <blocks, threads >> > (corr, var, kernelW, kernelH, dataW, dataH, fftW, fftH, result_mean, result_sd);
	sum_and_squaresum(corr, kernelW, kernelH, dataW, dataH, fftW, fftH, result_mean, result_sd, buffer);
}*/

/*__global__ void normalized_custompadding_sigma(float* corr, float* var, int kernelW, int kernelH, int dataW, int dataH, int fftW, int fftH, float* result_mean, float* result_sd) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = index / fftW;
	int index_x = index % fftW;
	if (index_y >= (dataH + kernelH - 1)) {
		if (index_y < fftH) {
			corr[index_y * fftW + index_x] = 0.0; //get rid of region we don't want
		}
		return;
	}
	if (index_x < (dataW + kernelW - 1)) {
		//corr[index] = (corr[index] - mean * sum[index]) / var[index];
		corr[index_y * fftW + index_x] = corr[index_y * fftW + index_x] / var[index_y * (dataW + kernelW - 1) + index_x];
		//in case of zero window variance
		if (var[index_y * (dataW + kernelW - 1) + index_x] <= powf(10, -8)) {
			corr[index_y * fftW + index_x] = 0;
		}
	}
	else {
		corr[index_y * fftW + index_x] = -0.0; //get rid of region we don't want
	}

}
**/
/*
void sum_and_squaresum(float* corr, int kernelW, int kernelH, int dataW, int dataH, int fftW, int fftH, float* result_mean, float* result_sd, float* buffer) {
	int n = fftH * fftW;
	int threads = 1024;
	int blocks = n / 2048 + 1 > 64 ? 64 : n / 1024 + 1;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6<float, 1024, true> << < dimGrid, dimBlock, smemSize >> > (corr, buffer, 1024 * 1024);

	n = blocks;
	threads = 512;
	blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6<float, 512, true> << < dimGrid, dimBlock, smemSize >> > (buffer, result_mean, 1024 * 1);

	n = 1024 * 1024;
	threads = 512;
	blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6_square<float, 512, true> << < dimGrid, dimBlock, smemSize >> > (corr, buffer, 1024 * 1024);

	n = 1024 * 1;
	threads = 512;
	blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6<float, 512, true> << < dimGrid, dimBlock, smemSize >> > (buffer, result_sd, 1024 * 1);
}
*/
void reduce6_max_call(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, unsigned int n) {
	int threads = 1024;
	//int blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	int blocks = 1024;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	//bool ispow2 = (bool)n && (!(n & (n - 1)));

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	reduce6_max<float, 1024, true> << < dimGrid, dimBlock, smemSize >> > (g_idata, g_odata, x_loc, n);


	threads = 512;
	blocks = 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	reduce6_max<float, 512, true> << < dimGrid, dimBlock, smemSize >> > (g_odata, g_odata1, y_loc, 1024);

}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_max(T* g_idata, T* g_odata, int* x_loc, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T myMax = -99999.0;
	int myMaxx = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		myMax = g_idata[i] > myMax ? g_idata[i] : myMax;
		myMaxx = tid;
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			//myMax = maxf(g_idata[i + blockSize], myMax);
			if (myMax < g_idata[i + blockSize]) {
				myMax = g_idata[i + blockSize];
				myMaxx += blockSize; //because blocksize = 256 and max threadid = 255
			}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	cg::sync(cta);

	if ((blockSize >= 2048) && (tid < 1024))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 256];
		if (myMax < sdata[tid + 1024]) {
			sdata[tid] = myMax = sdata[tid + 1024];
			myMaxx += 1024;
		}
	}

	if ((blockSize >= 1024) && (tid < 512))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 256];
		if (myMax < sdata[tid + 512]) {
			sdata[tid] = myMax = sdata[tid + 512];
			myMaxx += 512;
		}
	}

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 256];
		if (myMax < sdata[tid + 256]) {
			sdata[tid] = myMax = sdata[tid + 256];
			myMaxx += 256;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 128];

		if (myMax < sdata[tid + 128]) {
			sdata[tid] = myMax = sdata[tid + 128];
			myMaxx += 128;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 64]) {
			sdata[tid] = myMax = sdata[tid + 64];
			myMaxx += 64;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 64) && (tid < 32))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 32]) {
			sdata[tid] = myMax = sdata[tid + 32];
			myMaxx += 32;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 32) && (tid < 16))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 16]) {
			sdata[tid] = myMax = sdata[tid + 16];
			myMaxx += 16;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 16) && (tid < 8))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 8]) {
			sdata[tid] = myMax = sdata[tid + 8];
			myMaxx += 8;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 8) && (tid < 4))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 4]) {
			sdata[tid] = myMax = sdata[tid + 4];
			myMaxx += 4;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 4) && (tid < 2))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 2]) {
			sdata[tid] = myMax = sdata[tid + 2];
			myMaxx += 2;
		}
	}

	cg::sync(cta);

	if ((blockSize >= 2) && (tid < 1))
	{
		//sdata[tid] = myMax = myMax + sdata[tid + 64];

		if (myMax < sdata[tid + 1]) {
			sdata[tid] = myMax = sdata[tid + 1];
			myMaxx += 1;
		}
	}

	cg::sync(cta);

	//cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	//if (cta.thread_rank() < 32)
	//{
	//	// Fetch final intermediate sum from 2nd warp
	//	if (blockSize >= 64) {
	//		//myMax += sdata[tid + 32];
	//		if (myMax < sdata[tid + 32]) {
	//			myMax = sdata[tid + 32];
	//			myMaxx += 32;
	//		}
	//	}
	//	// Reduce final warp using shuffle
	//	for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
	//	{
	//		//myMax += tile32.shfl_down(myMax, offset);
	//		if (myMax < tile32.shfl_down(myMax, offset)) {
	//			myMax = tile32.shfl_down(myMax, offset);
	//			myMaxx += offset;
	//		}
	//	}
	//}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) {
		g_odata[blockIdx.x] = myMax;
		x_loc[blockIdx.x] = myMaxx;
		//y_loc[blockIdx.x] = myMaxy;
	}
}

void reduce3_max_call(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, unsigned int n) {
	int threads = 1024;
	//int blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	int blocks = 1024;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	//bool ispow2 = (bool)n && (!(n & (n - 1)));

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);
	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_idata, g_odata, x_loc, n);

	//std::cout << "test" << std::endl;

	threads = 512;
	blocks = 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);

	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_odata, g_odata1, y_loc, 1024);

}

void reduce3_max_custompadding_call(float* g_idata, float* g_odata, float* g_odata1, int* x_loc, int* y_loc, unsigned int n) {
	int threads = 512;
	//int blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	int blocks = 1024;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	//bool ispow2 = (bool)n && (!(n & (n - 1)));

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);
	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_idata, g_odata, x_loc, n);

	//std::cout << "test" << std::endl;

	threads = 512;
	blocks = 1;
	dimBlock = dim3(threads, 1, 1);
	dimGrid = dim3(blocks, 1, 1);
	smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);

	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_odata, g_odata1, y_loc, 1024);

}

template <class T>
__global__ void
reduce3_max(T* g_idata, T* g_odata, int* x_loc, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();
	__shared__ int indices[1024];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	int myMaxx = tid;
	T myMax = (i < n) ? g_idata[i] : 0;
	if (myMax == g_idata[i]) myMaxx = tid;

	if (i + blockDim.x < n)
		if (myMax < g_idata[i + blockDim.x]) {
			myMax = g_idata[i + blockDim.x];
			myMaxx = tid + blockDim.x;
		}

	sdata[tid] = myMax;
	indices[tid] = myMaxx;
	cg::sync(cta);

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (myMax < sdata[tid + s]) {
				myMax = sdata[tid + s];
				myMaxx = indices[tid + s];
			}
			sdata[tid] = myMax;
			indices[tid] = myMaxx;
			//sdata[tid] = mySum = mySum + sdata[tid + s];
		}

		cg::sync(cta);
	}

	// write result for this block to global mem
	if (tid == 0) {
		g_odata[blockIdx.x] = myMax;
		x_loc[blockIdx.x] = myMaxx;
	}
}
/*
void padDataClampToBorder_custom_call(float* d_PaddedData, float* d_Data, int fftH, int fftW, int dataH, int dataW, float mean) {
	int total_size = fftH * fftW;
	int threadnum = 1024;
	int blocknum = total_size / threadnum;
	padDataClampToBorder_custom<<<blocknum, threadnum >>> (d_PaddedData, d_Data, fftH, fftW, dataH, dataW, mean);
}

__global__ void padDataClampToBorder_custom(float* d_PaddedData, float* d_Data, int fftH, int fftW, int dataH, int dataW, float mean) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int indexx = index% fftW;
	int indexy = index / fftW;
	if (indexy < dataH && indexx < dataW) {
		d_PaddedData[index] = d_Data[indexy * dataW + indexx];
	}
	else {
		d_PaddedData[index] = mean;
	}
}

void padKernel_custom_call(float* d_PaddedKernel, float* d_Kernel, int fftH, int fftW, int kernelH, int kernelW) {
	int total_size = fftH * fftW;
	int threadnum = 1024;
	int blocknum = total_size / threadnum;
	padKernel_custom << <blocknum, threadnum >> > (d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW);
}

__global__ void padKernel_custom(float* d_PaddedKernel, float* d_Kernel, int fftH, int fftW, int kernelH, int kernelW) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int indexx = index % fftW;
	int indexy = index / fftW;
	if (indexx < kernelW && indexy < kernelH) {
		d_PaddedKernel[index] = d_Kernel[indexy * kernelW + indexx];
	}
	else {
		d_PaddedKernel[index] = 0.0;
	}
}
*/

__global__ void test_yz_call(float* data, float* val, int* loc, int size) {
	float max = -999999;
	*loc = 0;
	for (int i = 0; i < size; i++) {
		float d = data[i];
		if (d > max) {
			max = d;
			*loc = i;
		}
	}
	*val = max;
}

void yz_max(float* g_idata, float* final_data, int* final_index, float* buffer_data, int* buffer_index, int fftH, int fftW) {
	//a method that first use reduce3_max to group together the max val and loc for each block of 2048 elements then use single thread navie method to find max val and loc among them.
	int threads = 1024;
	int n = fftH * fftW;
	//int blocks = n / 1024 + 1 > 64 ? 64 : n / 1024 + 1;
	int blocks = n / 2048;
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	//bool ispow2 = (bool)n && (!(n & (n - 1)));

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);
	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_idata, buffer_data, buffer_index, n);

	test_yz_call << <1, 1 >> > (buffer_data, final_data, final_index, blocks);
}
