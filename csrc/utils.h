#include <stdio.h>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                  \
  if (((T2).size(0) != (T1).size(0)) || ((T2).size(1) != (T1).size(1)) || \
      ((T2).size(2) != (T1).size(2)) || ((T2).size(3) != (T1).size(3))) { \
    throw std::runtime_error("Tensor size mismatch!");                    \
  }

#define CUTE_PRINT(name, content) \
  print(name);                    \
  print(" : ");                   \
  print(content);                 \
  print("\n");

#define CUTE_PRINTTENSOR(name, content) \
  print(name);                          \
  print(" : ");                         \
  print_tensor(content);                \
  print("\n");
