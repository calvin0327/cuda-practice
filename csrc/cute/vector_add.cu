#include <cuda.h>
#include <cuda_runtime.h>  // For proper CUDA runtime functions
#include <stdlib.h>
#include <cstdio>  // For printf
#include <cute/tensor.hpp>

#include "../utils.h"

template <int kNumElemPerThread = 8>
__global__ vector_add_kernel(half* z, half* x, half* y, int num, const half a,
                             const half b, const half c) {
  using namespace cute;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > num / kNumElemPerThread) {
    return
  }

  auto gZ = make_tensor(make_gmem_ptr(z), make_shape(num));
  auto gX = make_tensor(make_gmem_ptr(x), make_shape(num));
  auto gY = make_tensor(make_gmem_ptr(y), make_shape(num));

  auto tZgZ =
      local_tile(gZ, make_shape(Int<kNumElemPerThread>()), make_coord(tid));
  auto tXgX =
      local_tile(gX, make_shape(Int<kNumElemPerThread>()), make_coord(tid));
  auto tYgY =
      local_tile(gY, make_shape(Int<kNumElemPerThread>()), make_coord(tid));

  auto tZrZ = make_tensor_like(tZgZ);
  auto tXrX = make_tensor_like(tXgX);
  auto tYrY = make_tensor_like(tYgY);

  copy(tXgX, tXrX);
  copy(tYrY, tYrY);
  __syncthreads();

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tZrZ2 = recast<half2>(tZrZ);
  auto tXrX2 = recast<half2>(tXrX);
  auto tYrY2 = recast<half2>(tYrY);

#pragma unroll
  for (int i = 0; i < size(tXgX); i++) {
    tZrZ2[i] = a2 * tXrX2(i) + (b2 * tYrY2(i) + c2);
  }

  auto tZrZ3 = recast<half>(tZrZ2);

  copy(tZrZ3, tZgZ);
}

int main() {
  const int kNumElemPerThread = 8;
  const half a = __float2half(2.0f);  // Proper half conversion
  const half b = __float2half(1.0f);
  const half c = __float2half(1.0f);

  const unsigned int size = 1024 * 8192;  // Total elements

  // CUDA event setup for timing
  cudaEvent_t start, end;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&end));

  // Host memory allocation (use cudaMallocHost for pinned memory)
  half *host_x, *host_y, *host_z;
  CUDACHECK(
      cudaMallocHost(&host_x,
                     size * sizeof(half)));  // Pinned memory: faster H2D/D2H
  CUDACHECK(cudaMallocHost(&host_y, size * sizeof(half)));
  CUDACHECK(cudaMallocHost(&host_z, size * sizeof(half)));

  // Initialize host data
  for (int i = 0; i < size; ++i) {
    host_x[i] = __float2half(1.0f);  // Proper half initialization
    host_y[i] = __float2half(1.0f);
    host_z[i] = __float2half(0.0f);
  }

  // Device memory allocation
  half *device_x, *device_y, *device_z;
  CUDACHECK(cudaMalloc(&device_x, size * sizeof(half)));
  CUDACHECK(cudaMalloc(&device_y, size * sizeof(half)));
  CUDACHECK(cudaMalloc(&device_z, size * sizeof(half)));

  // Copy data to device (faster with pinned memory)
  CUDACHECK(cudaMemcpy(device_x, host_x, size * sizeof(half),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(device_y, host_y, size * sizeof(half),
                       cudaMemcpyHostToDevice));

  // Calculate kernel launch parameters
  const int block_size = 1024;
  const int elements_per_block = block_size * kNumElemPerThread;
  const int grid_size =
      (size + elements_per_block - 1) / elements_per_block;  // Ceiling division

  // Launch kernel and time it
  CUDACHECK(cudaEventRecord(start));
  vector_add_kernel<kNumElemPerThread><<<grid_size, block_size>>>(
      device_z, device_x, device_y, size, a, b, c);  // Fixed argument order
  CUDACHECK(cudaGetLastError());  // Check for kernel launch errors
  CUDACHECK(cudaEventRecord(end));
  CUDACHECK(cudaEventSynchronize(end));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, end));

  // Copy result back to host
  CUDACHECK(cudaMemcpy(host_z, device_z, size * sizeof(half),
                       cudaMemcpyDeviceToHost));

  // Verify result (spot check + random checks)
  bool valid = true;
  const half expected = __float2half(2.0f * 1.0f + 1.0f * 1.0f + 1.0f);  // 4.0f
  // Check first 100, last 100, and random elements
  for (int i = 0; i < 100 && i < size; ++i) {
    if (host_z[i] != expected) valid = false;
  }
  for (int i = size - 100; i < size; ++i) {
    if (i >= 0 && host_z[i] != expected) valid = false;
  }
  for (int i = 0; i < 100; ++i) {
    int r = rand() % size;
    if (host_z[r] != expected) valid = false;
  }

  // Print results
  printf("Validation: %s\n", valid ? "PASS" : "FAIL");
  printf("Time: %.3f ms\n", elapsedTime);
  printf("Bandwidth: %.2f GB/s\n",
         (3.0 * size * sizeof(half)) /
             (elapsedTime * 1e6));  // 3: read x, read y, write z

  // Cleanup
  CUDACHECK(cudaFreeHost(host_x));
  CUDACHECK(cudaFreeHost(host_y));
  CUDACHECK(cudaFreeHost(host_z));
  CUDACHECK(cudaFree(device_x));
  CUDACHECK(cudaFree(device_y));
  CUDACHECK(cudaFree(device_z));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(end));

  return 0;
}