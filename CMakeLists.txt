cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(cuda-practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========================================================
# 1. Auto-detect PyTorch ABI version
# =========================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Detecting PyTorch _GLIBCXX_USE_CXX11_ABI: ${TORCH_CXX11_ABI}")

# Add definition based on detection result
if("${TORCH_CXX11_ABI}" STREQUAL "1")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# =========================================================
# 2. Find dependencies
# =========================================================
find_package(CUDAToolkit REQUIRED)

# Get torch cmake path
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PATH})
find_package(Torch REQUIRED)

message(STATUS "Found PyTorch: ${Torch_VERSION}")

# =========================================================
# 3. Find source files and generate executables
# =========================================================
file(GLOB CUTE_CUDA_SOURCES "${CMAKE_SOURCE_DIR}/csrc/cute/*.cu")

if(NOT CUTE_CUDA_SOURCES)
  message(FATAL_ERROR "No CUDA sources found under csrc/cute")
endif()

foreach(CU_FILE ${CUTE_CUDA_SOURCES})
  get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
 
  add_executable(${EXE_NAME} ${CU_FILE})

  target_include_directories(${EXE_NAME}
    PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      ${TORCH_INCLUDE_DIRS}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${Python3_INCLUDE_DIRS}
  )
  
  # Link libraries
  target_link_libraries(${EXE_NAME}
    PRIVATE
      ${TORCH_LIBRARIES}
      CUDA::cudart
  )

  target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
  
  # Compile options
  target_compile_options(${EXE_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
      --generate-code=arch=compute_80,code=sm_80
      --expt-relaxed-constexpr 
      -O3
      -use_fast_math
    >
  )

  set_target_properties(${EXE_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

endforeach()

message(STATUS "Generated executables for the following CUDA files:")
foreach(CU_FILE ${CUTE_CUDA_SOURCES})
  get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
  message(STATUS "  - ${EXE_NAME}")
endforeach()