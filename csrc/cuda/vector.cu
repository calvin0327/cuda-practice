#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void vectorAdd(const float* a, const float* b, float* c,
                          int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int numElements = 50000;
  const size_t size = numElements * sizeof(float);

  float* h_a = (float*)malloc(size);
  float* h_b = (float*)malloc(size);
  float* h_c = (float*)malloc(size);

  for (int i = 0; i < numElements; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (numElements + blockSize - 1) / blockSize;

  vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, numElements);
  cudaDeviceSynchronize();
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < numElements; i++) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
      ok = false;
      break;
    }
  }

  if (ok)
    printf("success\n");
  else
    printf("failed\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}