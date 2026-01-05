cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(cuda-practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find cuda dir
find_package(CUDAToolkit REQUIRED)
# message(STATUS "cuda headers dir: ${CUDAToolkit_INCLUDE_DIRS}")
# message(STATUS "cuda library dir: ${CUDAToolkit_LIBRARY_DIR}")

# find Python (required for torch/extension.h)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# find torch dir
set(PYTORCH_CMAKE_PATH "${PROJECT_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PATH})
find_package(Torch REQUIRED)
# message(STATUS "Found PyTorch: ${Torch_VERSION}")
# message(STATUS "PyTorch Include Dirs: ${Torch_INCLUDE_DIRS}")
# message(STATUS "PyTorch Libraries: ${Torch_LIBRARIES}")

file(GLOB CUTE_CUDA_SOURCES
  "${CMAKE_SOURCE_DIR}/csrc/cute/*.cu"
)

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
  
  target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
  
  set_target_properties(${EXE_NAME} PROPERTIES
    # CUDA_ARCHITECTURES "70;75;80;86;89"
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
  )

endforeach()

message(STATUS "Generated executables for the following CUDA files:")
foreach(CU_FILE ${CUTE_CUDA_SOURCES})
  get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
  message(STATUS "  - ${EXE_NAME}")
endforeach()