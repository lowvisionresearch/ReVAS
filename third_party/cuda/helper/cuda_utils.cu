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
		
#define THREADS_PER_BLOCK 1024 // assume compute > 2.0, if changed must also change in cuda_match.cpp 
           
// normalize data by normnum*invvars, get max and second max values and argmax
void normed_argmax(float *data, float *invvars, float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes, int fftH_x_fftW, float normnum, bool copyMap){
    
    int nblocks = fftH_x_fftW/THREADS_PER_BLOCK + (fftH_x_fftW % THREADS_PER_BLOCK != 0);            
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(nblocks);

    // normalization and per-block reduction
	norm_kernel << < dimGrid, dimBlock>> > (data, invvars, blockmaxes, blocksecondmaxes, blockargmaxes, fftH_x_fftW, normnum, copyMap);
    
    // between-block reduction
    argmax_kernel << <1, 1>> > (blockmaxes, blocksecondmaxes, blockargmaxes, nblocks);
    
    return;

} // end of normed_max
        
// normalize and intialize for argmax
__global__ void norm_kernel(float *data, float *invvars, float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes, int fftH_x_fftW, float normnum, bool copyMap){
   
    __shared__ float peaks[THREADS_PER_BLOCK];
    __shared__ float secondpeaks[THREADS_PER_BLOCK];
    __shared__ int argmaxes[THREADS_PER_BLOCK];    

    // each thread loads an element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= fftH_x_fftW)
        return;
    peaks[tid] = data[i]*invvars[i]*normnum;
    secondpeaks[tid] = 0;
    argmaxes[tid] = i;
    // if the output map is going to be copied, write normalized result back into global mem (slow)
    if (copyMap){
        data[i] = peaks[tid];
    }
    __syncthreads();

    // do per-block reduction
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid+s<fftH_x_fftW && tid<s){
            if (peaks[tid+s]>peaks[tid]){
                secondpeaks[tid] = max(peaks[tid],secondpeaks[tid+s]);
                peaks[tid] = peaks[tid+s];
                argmaxes[tid] = argmaxes[tid+s];
            }
            else{
                secondpeaks[tid] = max(peaks[tid+s],secondpeaks[tid]);
            }
        }
    }
    __syncthreads();

    if (tid==0) {
        blockmaxes[blockIdx.x] = peaks[0];
        blocksecondmaxes[blockIdx.x] = secondpeaks[0];
        blockargmaxes[blockIdx.x] = argmaxes[0];
    }
}
    
// get argmax
__global__ void argmax_kernel(float *blockmaxes, float *blocksecondmaxes, int *blockargmaxes, int nblocks){
    
    float maxval = -1;
    float secondval = -1;
    int argmax = -1;
    for (unsigned int i = 0; i < nblocks; i++){
        if (blockmaxes[i] > maxval){
            secondval = max(maxval,blocksecondmaxes[i]);
            maxval = blockmaxes[i]; 
            argmax = blockargmaxes[i];
        }
    }
   
    blockmaxes[0] = maxval;
    blocksecondmaxes[0] = secondval;
    blockargmaxes[0] = argmax;

}
        
// get FFT size from data size        
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
		
// print matrix
void printmat(float * devicemat, int H, int W, char * name) {
    
    float * localmat = (float *)malloc(H * W * sizeof(float));
    checkCudaErrors(cudaMemcpy(localmat, devicemat, H * W * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << name << " = [";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < (W / 2 + 1); j++) {
            std::cout << localmat[i + j*H]; 
            if (j != (W / 2 + 1)-1)
                std::cout << ", ";
        }
        if (i != H-1)
            std::cout << ";";
    }
    std::cout << "];" << std::endl;

    free(localmat);
}

// print complex matrix
void printcmat(fComplex * devicemat, int H, int W, char * name) {
    
    fComplex * localmat = (fComplex *)malloc(H * W * sizeof(fComplex));
    checkCudaErrors(cudaMemcpy(localmat, devicemat, H * W * sizeof(fComplex), cudaMemcpyDeviceToHost));
    
    std::cout << name << " = [";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < (W / 2 + 1); j++) {
            std::cout << localmat[i + j*H].x << "+" << localmat[i + j*H].y <<"*i" ; 
            if (j != (W / 2 + 1)-1)
                std::cout << ", ";
        }
        if (i != H-1)
            std::cout << ";";
    }
    std::cout << "];" << std::endl;

    free(localmat);
}
















#include <cooperative_groups.h>
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

	bool ispow2 = (bool)n && (!(n & (n - 1)));

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : 2 * threads * sizeof(float);
	reduce3_max<float> << < dimGrid, dimBlock, smemSize >> > (g_idata, buffer_data, buffer_index, n);

	test_yz_call << <1, 1 >> > (buffer_data, final_data, final_index, blocks);
}