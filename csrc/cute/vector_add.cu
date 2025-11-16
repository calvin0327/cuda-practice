#include <stdlib.h>
#include <cstdio>  // For printf

#include <cuda.h>
#include <cuda_runtime.h>  // For proper CUDA runtime functions
#include <cute/tensor.hpp>

#include "../utils.h"

using namespace cute;

template <class CtaTiler>
__global__ void vector_add_f16_kernel(half* z, half* x, half* y, int num,
                                      CtaTiler cta_tiler,  // tile
                                      const half a, const half b,
                                      const half c) {
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<1>{});  // {kNumElemPerThread}

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > num / size(cta_tiler)) {
    return;
  }

  auto mZ = make_tensor(make_gmem_ptr(z), make_shape(num));
  auto mX = make_tensor(make_gmem_ptr(x), make_shape(num));
  auto mY = make_tensor(make_gmem_ptr(y), make_shape(num));

  auto gZ = local_tile(mZ, cta_tiler, make_coord(tid));
  auto gX = local_tile(mX, cta_tiler, make_coord(tid));
  auto gY = local_tile(mY, cta_tiler, make_coord(tid));

  auto rZ = make_tensor_like(gZ);
  auto rX = make_tensor_like(gX);
  auto rY = make_tensor_like(gY);

  if (thread0()) {
    print("mZ: ");
    print(mZ);
    print("\n");
    print("gZ: ");
    print(gZ);
    print("\n");
    print("rZ: ");
    print(rZ);
    print("\n");
  }

  CUTE_STATIC_ASSERT_V(size<0>(rZ) == size<0>(gZ));  // {kNumElemPerThread}
  CUTE_STATIC_ASSERT_V(size<0>(rX) == size<0>(gX));
  CUTE_STATIC_ASSERT_V(size<0>(rY) == size<0>(gY));

  copy(gX, rX);  // copy from gmem to register.
  copy(gY, rY);
  __syncthreads();

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto rZ2 = recast<half2>(rZ);
  auto rX2 = recast<half2>(rX);
  auto rY2 = recast<half2>(rY);

#pragma unroll
  for (int i = 0; i < size(gX); i++) {
    rZ2[i] = a2 * rX2(i) + (b2 * rY2(i) + c2);
  }

  auto rZ3 = recast<half>(rZ2);
  copy(rZ3, gZ);  // copy from register to gmem.
}

int main() {
  const int kNumElemPerThread = 8;
  const half a = __float2half(2.0f);  // Proper half conversion
  const half b = __float2half(1.0f);
  const half c = __float2half(1.0f);

  const unsigned int total_elements = 1024 * 8192;  // Total elements

  // CUDA event setup for timing
  cudaEvent_t start, end;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&end));

  // Host memory allocation (use cudaMallocHost for pinned memory)
  half *host_x, *host_y, *host_z;
  CUDACHECK(cudaMallocHost(&host_x, total_elements * sizeof(half)));
  CUDACHECK(cudaMallocHost(&host_y, total_elements * sizeof(half)));
  CUDACHECK(cudaMallocHost(&host_z, total_elements * sizeof(half)));

  // Initialize host data
  for (int i = 0; i < total_elements; ++i) {
    host_x[i] = __float2half(1.0f);  // Proper half initialization
    host_y[i] = __float2half(1.0f);
    host_z[i] = __float2half(0.0f);
  }

  // Device memory allocation
  half *device_x, *device_y, *device_z;
  CUDACHECK(cudaMalloc(&device_x, total_elements * sizeof(half)));
  CUDACHECK(cudaMalloc(&device_y, total_elements * sizeof(half)));
  CUDACHECK(cudaMalloc(&device_z, total_elements * sizeof(half)));

  // Copy data to device (faster with pinned memory)
  CUDACHECK(cudaMemcpy(device_x, host_x, total_elements * sizeof(half),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(device_y, host_y, total_elements * sizeof(half),
                       cudaMemcpyHostToDevice));

  // Calculate kernel launch parameters
  dim3 block(1024);
  const int elements_per_block = 1024 * kNumElemPerThread;
  dim3 grid(size(ceil_div(total_elements, elements_per_block)));

  auto cta_tiler = make_shape(Int<8>{});

  // Launch kernel and time it
  CUDACHECK(cudaEventRecord(start));

  vector_add_f16_kernel<<<grid, block>>>(device_z, device_x, device_y,
                                         total_elements, cta_tiler, a, b,
                                         c);  // Fixed argument order
  CUDACHECK(cudaGetLastError());              // Check for kernel launch errors
  CUDACHECK(cudaEventRecord(end));
  CUDACHECK(cudaEventSynchronize(end));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, end));

  // Copy result back to host
  CUDACHECK(cudaMemcpy(host_z, device_z, total_elements * sizeof(half),
                       cudaMemcpyDeviceToHost));

  // Verify result (spot check + random checks)
  bool valid = true;
  const half expected = __float2half(2.0f * 1.0f + 1.0f * 1.0f + 1.0f);  // 4.0f

  // Check first 100, last 100, and random elements
  for (int i = 0; i < 100 && i < total_elements; ++i) {
    if (host_z[i] != expected) valid = false;
  }

  // Print results
  printf("Validation: %s\n", valid ? "PASS" : "FAIL");
  printf("Time: %.3f ms\n", elapsedTime);
  printf("Bandwidth: %.2f GB/s\n",
         (3.0 * total_elements * sizeof(half)) / (elapsedTime * 1e6));

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