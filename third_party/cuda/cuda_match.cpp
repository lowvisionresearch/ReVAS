/*
 * Function for fast matching in the Fourier domain on GPU with CUDA 
 *
 * To use: initialize with 
 * cuda_match(paddedReference,invLocalVars,stripHeight,stripWidth,firstTime)
 *  called in cuda_prep(paddedReference,stripHeight,stripWidth,firstTime)
 *
 * Then match each strip with
 * [correlationMap,xPeak,yPeak,peakValue,secondPeakValue] = cuda_match(strip, copyMap);
 *  matches strip to latest initialized paddedReference
 *  copyMap (boolean flag) determines if the entire correlationMap is 
 *  copied from GPU. This allows more downstream processing but 
 *  reduces speed.
 *
 * E Alexander Nov 2019
 * based on code developed with Yi Zong, James Fong, and Jay Shenoy
 */

#include "mex.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>
        
#include "helper/helper_cuda.h"
#include "helper/helper_functions.h"

#include "helper/cuda_utils.cuh"
#include "helper/convolutionFFT2D_common.h"

# define THREADS_PER_BLOCK 1024 // assume compute > 2.0, if changed must also change in cuda_utils.cu

/*
 * persistent data
 */
float
*d_Data,
*d_PaddedData,
*h_Kernel,
*d_Kernel,
*d_PaddedKernel,
*d_InvLocalVars,
*d_LocalMaxes,
*d_LocalSecondMaxes;

fComplex
*d_DataSpectrum,
*d_KernelSpectrum;

cufftHandle
fftPlanFwd,
fftPlanInv;

int   kernelH, kernelW, kernelY, kernelX;
int   dataH, dataW, fftH, fftW;
float normnum;
int   *d_LocalArgMaxes, *h_argmax;
bool  firstTime;


/*
 * Major funtions: initialize, reinitialize, and perform matching
 */
// intialize: set constants, allocate memory, put FFT(ref) on GPU
void init(float *paddedReference, float *invLocalVars) {
    
    // sizes: power of two/multiple of 512 for FFT, kernel shifts to origin
    fftH = snapTransformSize(dataH);
    fftW = snapTransformSize(dataW);
    kernelY = fftH/2;
    kernelX = fftW/2;
    
    // allocate device memory and copy data over
    checkCudaErrors(cudaMalloc((void**)& d_Data, dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_Data, paddedReference, dataH * dataW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)& d_InvLocalVars, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_InvLocalVars, invLocalVars, fftH * fftW * sizeof(float), cudaMemcpyHostToDevice));   
    
    // Fourier transform the reference
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));
    checkCudaErrors(cudaMalloc((void**)& d_PaddedData, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
    padDataClampToBorder(d_PaddedData, d_Data, fftW, fftH, dataW, dataH,
                         kernelW, kernelH, kernelX, kernelY);
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    checkCudaErrors(cudaDeviceSynchronize());
    
    // allocate memory for strip
    h_Kernel = (float*)malloc(kernelH * kernelW * sizeof(float));
    checkCudaErrors(cudaMalloc((void**)& d_Kernel, kernelH * kernelW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_PaddedKernel, fftH * fftW * sizeof(float))); 
    checkCudaErrors(cudaMalloc((void**)& d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  
    // initialize for finding max
    int nblocks = (fftH*fftW)/THREADS_PER_BLOCK + ((fftH*fftW) % THREADS_PER_BLOCK != 0);
    checkCudaErrors(cudaMalloc((void**)& d_LocalMaxes, nblocks * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_LocalSecondMaxes, nblocks * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_LocalArgMaxes, nblocks * sizeof(int)));
    h_argmax = (int *)malloc(1*sizeof(int));
    
} // end of init

// reintialize: reset constants, do not reallocate memory, put FFT(ref) on GPU
void reinit(float *paddedReference, float *invLocalVars) {
 
    // rewrite data and vars to gpu
    checkCudaErrors(cudaMemcpy(d_Data, paddedReference, dataH * dataW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_InvLocalVars, invLocalVars, fftH * fftW * sizeof(float), cudaMemcpyHostToDevice));   
    
    // Fourier transform the reference
    checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
    padDataClampToBorder(d_PaddedData, d_Data, fftW, fftH, dataW, dataH,
                         kernelW, kernelH, kernelX, kernelY);
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    checkCudaErrors(cudaDeviceSynchronize());
    
} // end of reinit

// match: cross-correlate kernel to paddedReference using FFT convolution
void match(float *kernel, float *correlationMap, int *xPeak, int *yPeak,
           float *peakValue, float *secondPeakValue, bool copyMap){

    //copying data in reversed order to use convlution for xcorrelation
    double template_mean = 0.0;
    for (int i = 0; i < kernelH; i++) {
        for (int j = 0; j < kernelW; j++) {
            h_Kernel[kernelW * kernelH - 1 - (j + i * kernelW)] = kernel[j + i*kernelW];
            template_mean += h_Kernel[kernelW * kernelH - 1 - (j + i * kernelW)] / (double)(kernelW * kernelH);
        }
    }
   
    // subtract mean and calculate std dev for normalization
    double template_sd = 0.0;
    for (int i = 0; i < kernelH; i++) {
        for (int j = 0; j < kernelW; j++) {
            h_Kernel[kernelW * kernelH - 1 - (j + i * kernelW)] -= template_mean;
            template_sd += pow(h_Kernel[kernelW * kernelH - 1 - (j + i * kernelW)], 2.0);
        }
    }
    template_sd = pow(template_sd, 0.5); // ignore 1/sqrt(N) factor, appears elsewhere
    
    // move kernel to GPU, zero pad
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    padKernel(d_PaddedKernel, d_Kernel, fftW, fftH, kernelW, kernelH, kernelX, kernelY);
    
    // pointwise multiply in Fourier domain
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
    checkCudaErrors(cudaDeviceSynchronize());
    modulateAndNormalize(d_KernelSpectrum, d_DataSpectrum, fftH, fftW, 1);    
    
    //transform result to real space through fftinv
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex*)d_KernelSpectrum, (cufftReal*)d_PaddedKernel));
    checkCudaErrors(cudaDeviceSynchronize());
    
    // normalize with pre-computed local variances and get max
    normnum = 1.0f/template_sd; // 1/N factor split between sds
    normed_argmax(d_PaddedKernel, d_InvLocalVars,
                  d_LocalMaxes, d_LocalSecondMaxes, d_LocalArgMaxes,
                  fftH*fftW, normnum, copyMap);
    
    // output: peak, second peak for comparison, peak location
    checkCudaErrors(cudaMemcpy(peakValue, d_LocalMaxes, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(secondPeakValue, d_LocalSecondMaxes, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_argmax, d_LocalArgMaxes, 1 * sizeof(int), cudaMemcpyDeviceToHost)); 
    xPeak[0] = (*h_argmax+fftH/2)/fftH+1; // 1D to 2D index
    yPeak[0] = (*h_argmax+1+fftW/2)%(fftH)? (*h_argmax+1)%(fftH) : fftH; // 1D to 2D index
    xPeak[0] = (xPeak[0]+fftW/2)%fftH-1; //fftshift
    yPeak[0] = (yPeak[0]+fftW/2)%fftW; //fftshift
    
    std::cout << xPeak[0] << ", " << yPeak[0] << ": " << peakValue[0] << std::endl;
   
    // output: correlation map
    if (copyMap){
        checkCudaErrors(cudaMemcpy(correlationMap, d_PaddedKernel, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    }
    else {
        correlationMap[0] = 0;
    }
    
} // end of match

/*
 * Main function:
 *
 * initial call: cuda_match(paddedReference,invLocalVars,kernelH,kernelW)
 * subsequent calls: cuda_match(strip,copyMap), 
 *                   returns [correlationMap,xPeak,yPeak,peakValue,secondPeakValue]
 *                   correlationMap is only full map if copyMap is true (this is slow)
 *                   otherwise it is scalar 0
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    // initialization: set up GPU, store global quantities
    if(nrhs > 2){
        
        // unpack inputs
        float* paddedReference = (float *)mxGetData(prhs[0]);
        float* invLocalVars = (float *)mxGetData(prhs[1]);
        dataH = (int)mxGetM(prhs[0]);
        dataW = (int)mxGetN(prhs[0]);
        kernelH = (int)*mxGetPr(prhs[2]);
        kernelW = (int)*mxGetPr(prhs[3]);
        firstTime = (bool)*mxGetPr(prhs[4]);
        
        // initialize
        if (firstTime){
            init(paddedReference,invLocalVars);
            std::cout << "CUDA match initialized" << std::endl;
        } else {
            reinit(paddedReference,invLocalVars);
            std::cout << "CUDA match reinitialized" << std::endl;
        }
        
        return;
    }
    
    // match strip
    // input: is the entire correlation going to be copied?
    bool copyMap = (bool)*mxGetPr(prhs[1]);
    // output: make pointers to return data
    if (copyMap)
        plhs[0] = mxCreateNumericMatrix(fftH, fftW, mxSINGLE_CLASS, mxREAL);
    else 
        plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT16_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT16_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    plhs[4] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float *correlationMap = (float *)mxGetPr(plhs[0]);
    int *xPeak = (int *)mxGetPr(plhs[1]);
    int *yPeak = (int *)mxGetPr(plhs[2]);
    float *peakValue = (float *)mxGetPr(plhs[3]);
    float *secondPeakValue = (float *)mxGetPr(plhs[4]);
  
    // match strip
    match((float *)mxGetData(prhs[0]),correlationMap,xPeak,yPeak,peakValue,secondPeakValue,copyMap);
   
    return;
}