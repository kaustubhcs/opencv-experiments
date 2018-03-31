#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "math.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define CUDA_TIMING

unsigned char *input_gpu;
unsigned long long int *hist_gpu;

/*******************************************************/
/*                 CUDA Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    #endif
    return result;
}
                
// GPU kernel and functions
__global__ void kernel(unsigned char *input,
                       unsigned int height,
                       unsigned int width,
                       unsigned long long int *hist,
                       int low_thresh,
                       int high_thresh){
        
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;


    if ( x < width && y < height){
        int val = input[x*height+y]; // Access image pixels
if (val <= high_thresh && val >= low_thresh) { // Setting user thresholds



        atomicAdd(&hist[val],1); // Histogram Algorithm
}
    }
}

void transpose_img(unsigned char *in_mat,
                   unsigned int height, 
                   unsigned int width,
                   unsigned long long int *hist,
                   int low_thresh,
                   int high_thresh){
                         
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;
    
    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu    , size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&hist_gpu  , 256*sizeof(unsigned long long int)));
    
    checkCuda(cudaMemset(hist_gpu , 0 , 256*sizeof(unsigned long long int)));
                
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
                        in_mat, 
                        height*width*sizeof(char), 
                        cudaMemcpyHostToDevice));

    checkCuda(cudaDeviceSynchronize());
    
    // Execute algorithm
    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif
    
    // Kernel Call
    kernel<<<dimGrid, dimBlock>>>(input_gpu, height, width, hist_gpu, low_thresh, high_thresh);
    
    checkCuda(cudaDeviceSynchronize());



    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif
        
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(hist,
                        hist_gpu,
                        256*sizeof(unsigned long long int),
                        cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
    checkCuda(cudaFree(hist_gpu));
    checkCuda(cudaFree(input_gpu));

}