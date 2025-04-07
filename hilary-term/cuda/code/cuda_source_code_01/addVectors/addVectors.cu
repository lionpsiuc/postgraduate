#include "stdio.h"

__global__ void add_arrays_gpu(float* in1, float* in2, float* out, int Ntot) {
  int idx =
      blockIdx.x * blockDim.x + threadIdx.x; // Set unique ID for each thread
  if (idx < Ntot)
    out[idx] = in1[idx] + in2[idx];
}

int main() {

  // Initialise some variables
  float *a, *b, *c;
  float *a_d, *b_d, *c_d;
  int    N = 18;
  int    i;

  // Allocate memory on the CPU
  a = (float*) malloc(N * sizeof(float));
  b = (float*) malloc(N * sizeof(float));
  c = (float*) malloc(N * sizeof(float));

  // Allocate memory on the GPU
  cudaMalloc((void**) &a_d, sizeof(float) * N);
  cudaMalloc((void**) &b_d, sizeof(float) * N);
  cudaMalloc((void**) &c_d, sizeof(float) * N);

  // Create the data
  for (i = 0; i < N; i++) {
    a[i] = (float) 2 * i;
    b[i] = -(float) i;
  }

  // Copy data from CPU to GPU (i.e., host to device)
  cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  int  block_size = 8;       // Eight threads per block
  dim3 dimBlock(block_size); // Block dimension (i.e., 3D)
  dim3 dimGrid((N / dimBlock.x) +
               (!(N % dimBlock.x) ? 0 : 1));               // Grid dimension
  add_arrays_gpu<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N); // Launch kernel

  // Copy results from GPU to CPU (i.e., device to host)
  cudaMemcpy(c, c_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for (i = 0; i < N; i++) {
    printf("a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n", i, a[i], i, b[i], i,
           c[i]);
  }

  // Free memory
  free(a);
  free(b);
  free(c);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}
