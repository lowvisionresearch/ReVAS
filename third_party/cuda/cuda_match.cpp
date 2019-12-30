/*
 * Function for fast matching in the Fourier domain on GPU with CUDA 
 *
 * To use: initialize with 
 * cuda_match(paddedReference,localVars,stripHeight,stripWidth)
 *
 * Then match with
 * [correlationMap,xPeak,yPeak,peakValue,secondPeakValue] = cuda_match(strip, copyMap);
 *  matches strip to paddedReference
 *  copyMap (boolean flag) determines if the entire correlationMap is 
 *  copied from GPU. This allows more downstream processing but 
 *  reduces speed.
 *
 * E Alexander Nov 2019
 * based on code developed with Yi Zong, James Fong, and Jay Shenoy
 */

#include "mex.h"
#include "cublas_v2.h"
        
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>
        
#include "helper/helper_cuda.h"
#include "helper/helper_functions.h"

#include "helper/cuda_utils.cuh"
#include "helper/convolutionFFT2D_common.h"

#define DEBUG 0

/*
 * persistent data
 */
float
*d_Data,
*d_PaddedData,
*d_Kernel,
*d_PaddedKernel,
*d_LocalVars,
*d_PaddedVars,
*d_KernelMean,
*h_Kernel;

fComplex
*d_DataSpectrum,
*d_KernelSpectrum;

cufftHandle
fftPlanFwd,
fftPlanInv;

int kernelH, kernelW, kernelY, kernelX;
int   dataH, dataW, fftH, fftW;

cublasHandle_t cublasHandle;
int *argmax;

/*
 * Debug funtions
 */
// print matrix
void printmat(float * mat, int H, int W, char * name) {
    if (!DEBUG){
        return;
    }
    std::cout << name << " = [";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            std::cout << mat[H*j + i];
            if (j != W-1)
                std::cout << ", ";
        }
        if (i != H-1)
            std::cout << "; ";
    }
    std::cout << "];" << std::endl;
}

// print complex matrix
void printcmat(fComplex * mat, int H, int W, char * name) {
    if (!DEBUG){
        return;
    }
    std::cout << name << " = [";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < (W / 2 + 1); j++) {
            std::cout << mat[i + j*H].x << "+" << mat[i + j*H].y <<"*i" ; 
            if (j != (W / 2 + 1)-1)
                std::cout << ", ";
        }
        if (i != H-1)
            std::cout << ";";
    }
    std::cout << "];" << std::endl;
}

/*
 * Major funtions: initialize and perform matching
 */
// intialize: set constants, allocate memory, put FFT(ref) on GPU
void init(float *paddedReference, float *localVars) {
    
    // sizes: power of two for FFT, kernel shifts to origin
    fftH = snapTransformSize(dataH + kernelH - 1);
    fftW = snapTransformSize(dataW + kernelW - 1);
    kernelY = fftH/2;
    kernelX = fftW/2;

    // allocate device memory and copy data over
    checkCudaErrors(cudaMalloc((void**)& d_Data, dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_Data, paddedReference, dataH * dataW * sizeof(float), cudaMemcpyHostToDevice));

    ///
    float * h_Data = (float *)malloc(dataH * dataW * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_Data, d_Data, dataH * dataW * sizeof(float), cudaMemcpyDeviceToHost));
    printmat(h_Data,dataH,dataW,"h_Data");
    
    
    ///
    checkCudaErrors(cudaMalloc((void**)& d_LocalVars, (dataW - kernelW - 1) * (dataH - kernelH - 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_LocalVars, localVars, (dataW - kernelW - 1) * (dataH - kernelH - 1) * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)& d_PaddedVars, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedVars, 0, fftH * fftW * sizeof(float)));
    
    ///
    float * h_LocalVars = (float *)malloc((dataW - kernelW - 1) * (dataH - kernelH - 1) * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_LocalVars, d_LocalVars, (dataW - kernelW - 1) * (dataH - kernelH - 1) * sizeof(float), cudaMemcpyDeviceToHost));
    printmat(h_LocalVars,(dataH - kernelH - 1),(dataW - kernelW - 1),"h_LocalVars");
    padDataClampToBorder(d_PaddedVars, d_LocalVars, fftW, fftH, (dataW - kernelW - 1), (dataH - kernelH - 1),
                         kernelW, kernelH, kernelX, kernelY);
    
    ///
    float * h_PaddedVars = (float *)malloc(fftH * fftW * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_PaddedVars, d_PaddedVars, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    printmat(h_PaddedVars,fftH,fftW,"h_PaddedVars");
   
    
    // Fourier transform the reference
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));
    checkCudaErrors(cudaMalloc((void**)& d_PaddedData, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
    padDataClampToBorder(d_PaddedData, d_Data, fftW, fftH, dataW, dataH,
                         kernelW, kernelH, kernelX, kernelY);

    ///
    float * h_PaddedData = (float *)malloc(fftH * fftW * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_PaddedData, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    printmat(h_PaddedData,fftH,fftW,"h_PaddedData");
    
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    
    
    
    // allocate memory for strip
    h_Kernel = (float*)malloc(kernelH * kernelW * sizeof(float));
    checkCudaErrors(cudaMalloc((void**)& d_Kernel, kernelH * kernelW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& d_PaddedKernel, fftH * fftW * sizeof(float)));    
    checkCudaErrors(cudaMalloc((void**)& d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void**)& d_KernelMean, 1 * sizeof(float)));
  
    // initialize for finding max
    cublasCreate(&cublasHandle);
    argmax = (int *)malloc(1*sizeof(int));
    
} // end of init

// match: cross-correlate kernel to paddedReference using FFT convolution
void match(float *kernel, float *correlationMap, int *xPeak, int *yPeak,
           float *peakValue, float *secondPeakValue, bool copyMap){

    //copying data in reversed order to use convlution for xcorrelation
    double template_mean = 0.0;
    for (int i = 0; i < kernelH; i++) {
        for (int j = 0; j < kernelW; j++) {
            h_Kernel[kernelW * kernelH - 1 - (j + i * kernelW)] = kernel[j + i*kernelW];
            // calculate mean for normalization
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
    template_sd = pow(template_sd, 0.5);
    
    // move kernel to GPU, zero pad
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    padKernel(d_PaddedKernel, d_Kernel, fftW, fftH, kernelW, kernelH, kernelX, kernelY);
    
    if (DEBUG){
        std::cout << "fftW: " << fftW << ", fftH: " << fftH << ", kernelW: " << kernelW << ", kernelH: " << kernelH << ", kernelX: " << kernelX << ", kernelY: " << kernelY << std::endl; 
        float * h_PaddedKernel = (float *)malloc(fftH * fftW * sizeof(float));
        checkCudaErrors(cudaMemcpy(h_PaddedKernel, d_PaddedKernel, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
        printmat(h_PaddedKernel,fftH,fftW,"h_PaddedKernel");
    }
    
    // pointwise multiply in Fourier domain
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
    checkCudaErrors(cudaDeviceSynchronize());
    modulateAndNormalize(d_KernelSpectrum, d_DataSpectrum, fftH, fftW, 1);    
    
    //transform result to real space through fftinv
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex*)d_KernelSpectrum, (cufftReal*)d_PaddedKernel));
    checkCudaErrors(cudaDeviceSynchronize());
    
    // normalize with pre-computed local variances and get max
    //normalized_call(d_PaddedKernel, d_LocalVars, d_KernelMean, kernelH, kernelW, dataH, dataW, kernelY, kernelX, fftH, fftW);    
    //normed_max(d_PaddedKernel, d_LocalVars, fftH*fftW, peakValue, secondPeakValue, argmax);

    // get peak
    cublasIsamax(cublasHandle,fftH*fftW,d_PaddedKernel,1,argmax);
    checkCudaErrors(cudaMemcpy(peakValue, d_PaddedKernel+*argmax-1, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    //peakValue[0] = peakValue[0]/(pow(fftH*fftW,0.5)*template_sd);
    xPeak[0] = (*argmax+1+fftW/2)%(fftH)? (*argmax+1)%(fftH) : fftH;
    yPeak[0] = (*argmax+fftH/2)/fftH+1;
    xPeak[0] = (xPeak[0]+fftW/2)%fftW-1; //fftshift
    yPeak[0] = (yPeak[0]+fftW/2)%fftH-1; //fftshift

    // get second peak for comparison
    checkCudaErrors(cudaMemset(d_PaddedKernel+*argmax-1, 0, 1 * sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
    cublasIsamax(cublasHandle,fftH*fftW,d_PaddedKernel,1,argmax);
    checkCudaErrors(cudaMemcpy(secondPeakValue, d_PaddedKernel+*argmax-1, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    //secondPeakValue[0] = secondPeakValue[0]/(pow(fftH*fftW,0.5)*template_sd);

    // return correlationmap
    if (copyMap){
        // TODO: return peak
        // checkCudaErrors(cudaMemset(d_PaddedKernel+*argmax-1, peakValue[0], 1 * sizeof(float)));
        // checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(correlationMap, d_PaddedKernel, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    }
    else {
        correlationMap[0] = 0;
    }
    
    ///
    //std::cout << xPeak[0] << "," << yPeak[0] << std::endl;
    //std::cout << peakValue[0] << "," << template_sd  << std::endl;
    
} // end of match

/*
 * main function
 *
 * initial call: cuda_match(paddedReference,localVars,kernelH,kernelW)
 * subsequent calls: cuda_match(strip,copyMap), returns [correlationMap,xPeak,yPeak,peakValue,secondPeakValue]
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    // initialization: set up GPU, store global quantities
    if(nrhs > 2){
        
        // unpack inputs
        float* paddedReference = (float *)mxGetData(prhs[0]);
        float* localVars = (float *)mxGetData(prhs[1]);
        dataH = (int)mxGetM(prhs[0]);
        dataW = (int)mxGetN(prhs[0]);
        kernelH = (int)*mxGetPr(prhs[2]);
        kernelW = (int)*mxGetPr(prhs[3]);
        
        // initialize
        init(paddedReference,localVars);
        
        std::cout << "CUDA match initialized" << std::endl;
        
        return;
    }
    
    // match strip
    // input: is the entire correlation going to be copied
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