#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"

__global__ void softmax_kernel_v2(const float* input, float* output, int M,
                                  int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;  // 128 threads, every thread handles 32 elements

  const float* input_raw = input + bid * N;
  float* output_raw = output + bid * N;

  extern __shared__ float smem[];

  // pass 1: max
  float maxval = -1e20f;
  for (int i = tid; i < N; i += block_size) {
    maxval = fmaxf(maxval, input_raw[i]);
  }
  smem[tid] = maxval;
  __syncthreads();

  for (int stride = block_size / 2; stride > 32; stride /= 2) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* v = smem;
    v[tid] = fmaxf(v[tid], v[tid + 32]);
    v[tid] = fmaxf(v[tid], v[tid + 16]);
    v[tid] = fmaxf(v[tid], v[tid + 8]);
    v[tid] = fmaxf(v[tid], v[tid + 4]);
    v[tid] = fmaxf(v[tid], v[tid + 2]);
    v[tid] = fmaxf(v[tid], v[tid + 1]);
  }
  __syncthreads();
  maxval = smem[0];

  // pass 2: sum
  float sumval = 0.f;
  for (int i = tid; i < N; i += block_size) {
    sumval += expf(input_raw[i] - maxval);
  }
  smem[tid] = sumval;
  __syncthreads();

  for (int stride = block_size / 2; stride > 32; stride /= 2) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* v = smem;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
  }
  __syncthreads();
  sumval = smem[0];

  // pass 3: norm
  float norm = 1.f / sumval;
  for (int i = tid; i < N; i += block_size) {
    output_raw[i] = expf(input_raw[i] - maxval) * norm;
  }
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int size = M * N;

  float* h_inp = (float*)malloc(size * sizeof(float));
  float* h_out = (float*)malloc(size * sizeof(float));

  for (int i = 0; i < size; ++i) {
    h_inp[i] = (float)(rand() % 100) / 10.f;
  }

  float *d_inp, *d_out;
  cudaMalloc(&d_inp, size * sizeof(float));
  cudaMalloc(&d_out, size * sizeof(float));

  cudaMemcpy(d_inp, h_inp, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(M);                   // handle a row of elements peer block
  dim3 block((N + 32 - 1) / 32);  // handle 32 elements peer thread
  softmax_kernel_v2<<<grid, block>>>(d_inp, d_out, M, N);

  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

  printf("First row results (first 10 elements): \n");
  for (int i = 0; i < 10 && i < N; ++i) {
    printf("%.6f ", h_out[i]);
  }
  printf("\n");

  float row_sum = 0;
  for (int i = 0; i < N; ++i) {
    row_sum += h_out[i];
  }
  printf("First row sum: %.6f (expected: 1)\n", row_sum);

  delete[] h_inp;
  delete[] h_out;
  cudaFree(d_inp);
  cudaFree(d_out);
  return 0;
}
