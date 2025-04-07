#include <stdio.h>

int main() {
  int                   i, n, best, bestNumberOfMultiprocessors;
  int                   numberOfCUDAcoresForThisCC = 0;
  struct cudaDeviceProp x;

  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    printf("No CUDA-enabled devices were found\n");
  }
  printf("Found %d CUDA-enabled devices\n", n);
  best                        = -1;
  bestNumberOfMultiprocessors = -1;
  for (i = 0; i < n; i++) {
    cudaGetDeviceProperties(&x, i);
    printf("========================= IDENTITY DATA "
           "==================================\n");
    printf("GPU Model name: %s\n", x.name);
    if (x.integrated == 1) {
      printf("GPU The device is an integrated (motherboard) GPU\n");
    } else {
      printf("GPU The device is NOT an integrated (motherboard) GPU (i.e., it "
             "is a discrete device)\n");
    }
    printf("GPU pciBusID: %d\n", x.pciBusID);
    printf("GPU pciDeviceID: %d\n", x.pciDeviceID);
    printf("GPU pciDomainID: %d\n", x.pciDomainID);
    if (x.tccDriver == 1) {
      printf("The device is a Tesla one using TCC driver\n");
    } else {
      printf("The device is NOT a Tesla one using TCC driver\n");
    }
    printf("========================= COMPUTE DATA "
           "==================================\n");
    printf("GPU CC: %d.%d\n", x.major, x.minor);
    switch (x.major) {
      case 1: // Tesla or T10
        numberOfCUDAcoresForThisCC = 8;
        break;
      case 2: // Fermi
        switch (x.minor) {
          case 0: // 2.0
            numberOfCUDAcoresForThisCC = 32;
            break;
          case 1: // 2.1
            numberOfCUDAcoresForThisCC = 48;
            break;
          default: // Unknown
            numberOfCUDAcoresForThisCC = 0;
            break;
        }
        break;
      case 3: // Kepler
        numberOfCUDAcoresForThisCC = 192;
        break;
      case 5: // Maxwell
        numberOfCUDAcoresForThisCC = 128;
        break;
      case 6: // Pascal
        switch (x.minor) {
          case 0: // GP100, 64 CUDA cores per SM - 7.0 should be preferred
                  // over 7.1
            numberOfCUDAcoresForThisCC = 64;
            break;
          case 1: // GP102, GP104, GP106, GP107 - 128 CUDA cores per SM
          case 2: // GP10B, Pascal Tegra cards - still 128 CUDA cores per SM
            numberOfCUDAcoresForThisCC = 128;
            break;
          default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
            numberOfCUDAcoresForThisCC = 0;
            break;
        }
        break;
      case 7: // Volta is 7.0 and 7.2, 64 CUDA cores per SM, Turing is 7.5 -
              // also has 64 CUDA cores per SM
        numberOfCUDAcoresForThisCC = 64;
        break;
      case 8: // Ampere 8.x, with x < 9, has 64 CUDA cores per SM, but Ada
              // Lovelace (8.9) has 128 CUDA cores per SM
        switch (x.minor) {
          case 0: // The GA100 in the A100 is an Ampere (8.0) which has 64 CUDA
                  // cores per SM
            numberOfCUDAcoresForThisCC = 64;
            break;
          case 6: // The GeForce 3000 series is an Ampere (8.6) with 128 CUDA
                  // cores per SM
            numberOfCUDAcoresForThisCC = 128;
            break;
          case 9: // The GeForce 4000 series are Ada Lovelace (8.9) with 128
                  // CUDA cores per SM
            numberOfCUDAcoresForThisCC = 128;
            break;
          default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
            numberOfCUDAcoresForThisCC = 64;
            break;
        }
        break;
      case 9: // Hopper (G100 is 9.0) and Grace Hopper both have 128 CUDA cores
              // per SM
        numberOfCUDAcoresForThisCC = 128;
        break;
      case 10: // Blackwell has 128 CUDA cores per SM
        numberOfCUDAcoresForThisCC = 128;
        break;
      default: // Unknown
        numberOfCUDAcoresForThisCC = 0;
        break;
    }
    if (x.multiProcessorCount >
        bestNumberOfMultiprocessors * numberOfCUDAcoresForThisCC) {
      best = i;
      bestNumberOfMultiprocessors =
          x.multiProcessorCount * numberOfCUDAcoresForThisCC;
    }
    printf("GPU Clock frequency in Hz: %d\n", x.clockRate);
    printf("GPU Device can concurrently copy memory and execute a kernel: %d\n",
           x.deviceOverlap);
    printf("GPU Number of SMs: %d\n", x.multiProcessorCount);
    printf("GPU maximum number of threads per SM: %d\n",
           x.maxThreadsPerMultiProcessor);
    printf("GPU Maximum size of each dimension of a grid: %dx%dx%d\n",
           x.maxGridSize[0], x.maxGridSize[1], x.maxGridSize[2]);
    printf("GPU Maximum size of each dimension of a block: %dx%dx%d\n",
           x.maxThreadsDim[0], x.maxThreadsDim[1], x.maxThreadsDim[2]);
    printf("GPU Maximum number of threads per block: %d\n",
           x.maxThreadsPerBlock);
    printf("GPU Maximum pitch in bytes allowed by memory copies: %u\n",
           (unsigned int) (x.memPitch));
    printf("GPU Compute mode is: %d\n", x.computeMode);
    printf("========================= MEMORY DATA "
           "==================================\n");
    printf("GPU Total global memory: %zu bytes\n", (size_t) (x.totalGlobalMem));
    printf("GPU Peak memory clock frequency in kHz: %d \n", x.memoryClockRate);
    printf("GPU Memory bus width: %d bits\n", x.memoryBusWidth);
    printf("GPU L2 cache size: %d bytes\n", x.l2CacheSize);
    printf("GPU 32-bit registers available per block: %d\n", x.regsPerBlock);
    printf("GPU Shared memory available per block in bytes: %d\n",
           (int) (x.sharedMemPerBlock));
    printf("GPU Alignment requirement for textures: %d\n",
           (int) (x.textureAlignment));
    printf("GPU Constant memory available on device in bytes: %d\n",
           (int) (x.totalConstMem));
    printf("GPU Warp size in threads: %d\n", x.warpSize);
    printf("GPU Maximum 1D texture size: %d\n", x.maxTexture1D);
    printf("GPU Maximum 2D texture size: %d %d\n", x.maxTexture2D[0],
           x.maxTexture2D[1]);
    printf("GPU Maximum 3D texture size: %d %d %d\n", x.maxTexture3D[0],
           x.maxTexture3D[1], x.maxTexture3D[2]);
    printf("GPU Maximum 1D layered texture dimensions: %d %d\n",
           x.maxTexture1DLayered[0], x.maxTexture1DLayered[1]);
    printf("GPU Maximum 2D layered texture dimensions: %d %d %d\n",
           x.maxTexture2DLayered[0], x.maxTexture2DLayered[1],
           x.maxTexture2DLayered[2]);
    printf("GPU Surface alignment: %d\n", (int) (x.surfaceAlignment));
    if (x.canMapHostMemory == 1) {
      printf(
          "GPU The device can map host memory into the CUDA address space\n");
    } else {
      printf("GPU The device can NOT map host memory into the CUDA address "
             "space\n");
    }
    if (x.ECCEnabled == 1) {
      printf("GPU Memory has ECC support\n");
    } else {
      printf("GPU Memory does not have ECC support\n");
    }
    if (x.ECCEnabled == 1) {
      printf("GPU The device shares an unified address space with the host\n");
    } else {

      printf("GPU The device DOES NOT share an unified address space with the "
             "host\n");
    }
    printf("========================= EXECUTION DATA "
           "==================================\n");
    if (x.concurrentKernels == 1) {
      printf("GPU Concurrent kernels are allowed\n");
    } else {
      printf("GPU Concurrent kernels are NOT allowed\n");
    }
    if (x.kernelExecTimeoutEnabled == 1) {
      printf(
          "GPU There is a runtime limit for kernels executed in the device\n");
    } else {
      printf("GPU There is NOT a runtime limit for kernels executed in the "
             "device\n");
    }
    if (x.asyncEngineCount == 1) {
      printf("GPU The device can concurrently copy memory between host and "
             "device while executing a kernel\n");
    } else if (x.asyncEngineCount == 2) {
      printf(
          "GPU The device can concurrently copy memory between host and device "
          "in both directions and execute a kernel at the same time\n");
    } else {
      printf("GPU The device is NOT capable of concurrently copying memory\n");
    }
  }
  if (best >= 0) {
    cudaGetDeviceProperties(&x, best);
    printf("Choosing %s with %d CUDA cores\n", x.name,
           bestNumberOfMultiprocessors);
    cudaSetDevice(best);
  }
}
