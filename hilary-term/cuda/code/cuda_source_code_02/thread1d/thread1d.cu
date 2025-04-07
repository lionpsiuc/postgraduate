#include "stdio.h"

__global__ void scanTheadInformationGPU(int* threadIdsGPU, int* blockIdsGPU,
                                        int Ntot) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Ntot) {
    threadIdsGPU[idx] = threadIdx.x;
    blockIdsGPU[idx]  = blockIdx.x;
  }
}

int main() {
  int *threadIds, *blockIds;       // Pointers to host memory
  int *threadIdsGPU, *blockIdsGPU; // Pointers to device memory
  int  N = 18;                     // Total size that we want
  int  i;

  // Allocate arrays threadIds and blockIds on host
  threadIds = (int*) malloc(N * sizeof(int));
  blockIds  = (int*) malloc(N * sizeof(int));

  // Allocate arrays threadIdsGPU and blockIdsGPU on device
  cudaMalloc((void**) &threadIdsGPU, sizeof(int) * N);
  cudaMalloc((void**) &blockIdsGPU, sizeof(int) * N);

  // Compute the execution configuration
  int  block_size = 8;
  dim3 dimBlock(block_size);
  dim3 dimGrid((N / dimBlock.x) + (!(N % dimBlock.x) ? 0 : 1));

  // Scan information from the threads
  scanTheadInformationGPU<<<dimGrid, dimBlock>>>(threadIdsGPU, blockIdsGPU, N);

  // Copy data from device memory to host memory
  cudaMemcpy(threadIds, threadIdsGPU, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(blockIds, blockIdsGPU, sizeof(int) * N, cudaMemcpyDeviceToHost);

  // Print all the data about the threads
  printf("dimGrid = %d\n", dimGrid.x);
  for (i = 0; i < N; i++) {
    printf("threadIds[%d] = %d\n", i, threadIds[i]);
  }
  for (i = 0; i < N; i++) {
    printf("blockIds[%d] = %d\n", i, blockIds[i]);
  }

  // Free memory
  free(threadIds);
  free(blockIds);
  cudaFree(threadIdsGPU);
  cudaFree(blockIdsGPU);
}
