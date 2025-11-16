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
