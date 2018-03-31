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
unsigned char *output_gpu;

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
                       unsigned char *output,
                       unsigned int height,
                       unsigned int width){
        
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;


    if (x > 0 && x < width-1 && y > 0 && y < height-1 ){

        int loc = y*width+x;
        int i1 = ((int)input[loc-width-1]) * -1;
        int i2 = ((int)input[loc-width]) * -2;
        int i3 = ((int)input[loc-width+1]) * -1;
        int i4 = ((int)input[loc+width-1]) * 1;
        int i5 = ((int)input[loc+width]) * 2;
        int i6 = ((int)input[loc+width-1]) * 1;
        int it=0;
        it = (i1 + i2 + i3 + i4 + i5 + i6)/6;

        int d1 = ((int)input[loc-width-1]) * 1;
        int d2 = ((int)input[loc-1]) * 2;
        int d3 = ((int)input[loc-1]) * 1;
        int d4 = ((int)input[loc-width+1]) * -1;
        int d5 = ((int)input[loc+1]) * -2;
        int d6 = ((int)input[loc+width+1]) * -1;
        int dt=0;
        dt = (d1 + d2 + d3 + d4 + d5 + d6)/6;
        int total=0;
        total = (int)(sqrt((float)it*(float)it + (float)dt*(float)dt));
        output[loc] = (unsigned char)total;

}
}

void transpose_img(unsigned char *in_mat, 
                   unsigned char *out_mat, 
                   unsigned int height, 
                   unsigned int width){
                         
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;
    
    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu    , size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
    
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
                
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
    kernel<<<dimGrid, dimBlock>>>(input_gpu, output_gpu, height, width);
    
    checkCuda(cudaDeviceSynchronize());
    
    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif
        
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(out_mat, 
                        output_gpu, 
                        height*width*sizeof(unsigned char), 
                        cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
    checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));

}