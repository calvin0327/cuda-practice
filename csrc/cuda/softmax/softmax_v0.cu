#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"

__global__ void softmax_kernel_v0(const float* input, float* output, int M,
                                  int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M) {
    const float* input_row = input + tid * N;
    float* out_row = output + tid * N;

    // maxval
    float maxval = -INFINITY;
    for (int i = 0; i < N; ++i) {
      maxval = fmaxf(input_row[i], maxval);
    }

    // sum
    float sum = 0;
    for (int i = 0; i < N; ++i) {
      out_row[i] = expf(input_row[i] - maxval);
      sum += out_row[i];
    }

    // norm
    float norm = 0.1f / sum;
    for (int i = 0; i < N; ++i) {
      out_row[i] *= norm;
    }
  }
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int size = M * N;

  float* h_input = (float*)malloc(size * sizeof(float));
  float* h_output = (float*)malloc(size * sizeof(float));

  // float range from 0 to 10
  for (int i = 0; i < size; ++i) {
    h_input[i] = (float)(rand() % 100) / 10.f;
  }

  float *d_input, *d_output;
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));

  cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(128);
  dim3 block((M + 128 - 1) / 128);
  softmax_kernel_v0<<<grid, block>>>(d_input, d_output, M, N);

  cudaDeviceSynchronize();
  cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

  printf("First row results (first 10 elements): \n");
  for (int i = 0; i < 10 && i < N; ++i) {
    printf("%.6f ", h_output[i]);
  }
  printf("\n");

  float row_sum = 0;
  for (int i = 0; i < N; ++i) {
    row_sum += h_output[i];
  }
  printf("First row sum: %.6f (expected: 1)\n", row_sum);

  delete[] h_input;
  delete[] h_output;
  delete d_input;
  delete d_output;
  return 0;
}
