#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"

__device__ float warp_reduce_max(float max) {
  max = fmaxf(max, __shfl_xor_sync(0xFFFFFFFF, max, 16));
  max = fmaxf(max, __shfl_xor_sync(0xFFFFFFFF, max, 8));
  max = fmaxf(max, __shfl_xor_sync(0xFFFFFFFF, max, 4));
  max = fmaxf(max, __shfl_xor_sync(0xFFFFFFFF, max, 2));
  max = fmaxf(max, __shfl_xor_sync(0xFFFFFFFF, max, 1));
  return max;
}

__device__ float warp_reduce_sum(float sum) {
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
  return sum;
}

__global__ void softmax_kernel_v4(const float* input, float* output, int M,
                                  int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;  // 128 threads, every thread handles 32 elements

  int laneId = tid % 32;
  int warpId = tid / 32;
  int nums_warp = block_size / 32;

  const float4* input_raw = reinterpret_cast<const float4*>(input + bid * N);
  float4* output_raw = reinterpret_cast<float4*>(output + bid * N);

  extern __shared__ float smem[];

  // pass 1: max
  // every thread load 4 float evrery time
  float maxval = -1e20f;
  for (int i = tid; i < N / 4; i += block_size) {
    float4 v = input_raw[i];
    maxval = fmaxf(maxval, fmaxf(fmax(v.x, v.y), fmaxf(v.z, v.w)));
  }

  maxval = warp_reduce_max(maxval);
  if (laneId == 0) {
    smem[warpId] = maxval;
  }
  __syncthreads();

  maxval = tid < nums_warp ? smem[tid] : -1e20f;
  maxval = warp_reduce_max(maxval);
  if (tid == 0) {
    smem[0] = maxval;
  }
  __syncthreads();
  maxval = smem[0];

  // pass 2: sum
  float sumval = 0.f;
  for (int i = tid; i < N / 4; i += block_size) {
    float4 v = input_raw[i];
    sumval += expf(v.x - maxval) + expf(v.y - maxval) + expf(v.z - maxval) +
              expf(v.w - maxval);
  }

  sumval = warp_reduce_sum(sumval);
  if (laneId == 0) {
    smem[warpId] = sumval;
  }
  __syncthreads();

  sumval = tid < nums_warp ? smem[tid] : 0.f;
  sumval = warp_reduce_sum(sumval);
  if (tid == 0) {
    smem[0] = sumval;
  }
  __syncthreads();
  sumval = smem[0];

  // pass 3: norm
  float norm = 1.f / sumval;
  for (int i = tid; i < N / 4; i += block_size) {
    float4 v = input_raw[i];
    float4 o;

    o.x = expf(v.x - maxval) * norm;
    o.y = expf(v.y - maxval) * norm;
    o.z = expf(v.z - maxval) * norm;
    o.w = expf(v.w - maxval) * norm;

    output_raw[i] = o;
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

  dim3 grid(M);                     // handle a row of elements peer block
  dim3 block((N + 128 - 1) / 128);  // handle 32 elements peer thread
  softmax_kernel_v4<<<grid, block>>>(d_inp, d_out, M, N);

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
