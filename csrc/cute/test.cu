#include "cute/tensor.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

using T = cute::half_t;
using namespace cute;

int main() {
  thrust::host_vector<T> host_a(100);
  thrust::host_vector<T> host_b(100);
  thrust::host_vector<T> host_c(100);

  thrust::device_vector<T> device_a = host_a;
  thrust::device_vector<T> device_b = host_b;
  thrust::device_vector<T> device_c = host_c;

  return 0;
}