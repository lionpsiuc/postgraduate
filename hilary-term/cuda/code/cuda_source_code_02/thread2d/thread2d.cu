#include "stdio.h"

__global__ void scanTheadInformationGPU(int* threadXIdsGPU, int* threadYIdsGPU,
                                        int* blockXIdsGPU, int* blockYIdsGPU,
                                        int N, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < N) {
    if (idy < M) {
      threadXIdsGPU[idx + idy * N] = threadIdx.x;
      threadYIdsGPU[idx + idy * N] = threadIdx.y;
      blockXIdsGPU[idx + idy * N]  = blockIdx.x;
      blockYIdsGPU[idx + idy * N]  = blockIdx.y;
    }
  }
}

int main() {
  int **threadXIds, **threadYIds;
  int*  threadXIds1d = NULL;
  int*  threadYIds1d = NULL;
  int **blockXIds, **blockYIds;
  int*  blockXIds1d = NULL;
  int*  blockYIds1d = NULL;
  int * threadXIdsGPU, *threadYIdsGPU;
  int * blockXIdsGPU, *blockYIdsGPU;
  int   N = 3, M = 3;
  int   i, j;
  threadXIds1d = (int*) malloc((N) * (M) * sizeof(int));
  threadYIds1d = (int*) malloc((N) * (M) * sizeof(int));
  threadXIds   = (int**) malloc((N) * sizeof(int*));
  threadYIds   = (int**) malloc((N) * sizeof(int*));
  for (i = 0; i < N; i++) {
    threadXIds[i] = (&(threadXIds1d[i * M]));
    threadYIds[i] = (&(threadYIds1d[i * M]));
  }
  blockXIds1d = (int*) malloc((N) * (M) * sizeof(int));
  blockYIds1d = (int*) malloc((N) * (M) * sizeof(int));
  blockXIds   = (int**) malloc((N) * sizeof(int*));
  blockYIds   = (int**) malloc((N) * sizeof(int*));
  for (i = 0; i < N; i++) {
    blockXIds[i] = (&(blockXIds1d[i * M]));
    blockYIds[i] = (&(blockYIds1d[i * M]));
  }
  cudaMalloc((void**) &threadXIdsGPU, sizeof(int) * N * M);
  cudaMalloc((void**) &threadYIdsGPU, sizeof(int) * N * M);
  cudaMalloc((void**) &blockXIdsGPU, sizeof(int) * N * M);
  cudaMalloc((void**) &blockYIdsGPU, sizeof(int) * N * M);
  int  block_size = 2;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid((N / dimBlock.x) + (!(N % dimBlock.x) ? 0 : 1),
               (M / dimBlock.y) + (!(M % dimBlock.y) ? 0 : 1));
  scanTheadInformationGPU<<<dimGrid, dimBlock>>>(
      threadXIdsGPU, threadYIdsGPU, blockXIdsGPU, blockYIdsGPU, N, M);
  cudaMemcpy(threadXIds1d, threadXIdsGPU, sizeof(int) * N * M,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(threadYIds1d, threadYIdsGPU, sizeof(int) * N * M,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(blockXIds1d, blockXIdsGPU, sizeof(int) * N * M,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(blockYIds1d, blockYIdsGPU, sizeof(int) * N * M,
             cudaMemcpyDeviceToHost);
  printf(" dimGrid = %d %d\n", dimGrid.x, dimGrid.y);
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      printf(" threadIds[%d][%d]= %d , %d\n", i, j, threadXIds[i][j],
             threadYIds[i][j]);
    }
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      printf(" blockIds[%d][%d]= %d , %d\n", i, j, blockXIds[i][j],
             blockYIds[i][j]);
    }
  }
  free(threadXIds);
  free(threadXIds1d);
  free(threadYIds);
  free(threadYIds1d) free(blockXIds);
  free(blockXIds1d);
  free(blockYIds);
  free(blockYIds1d);
  cudaFree(threadXIdsGPU);
  cudaFree(threadYIdsGPU);
  cudaFree(blockXIdsGPU);
  cudaFree(blockYIdsGPU);
}
