#include <stdio.h>

#include <cuda.h>

#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

using namespace cute;

void fundamemtal_operations();
void nonowning_tensors();

__global__ void owning_tensors() {
  // Shared memory (static or dynamic layouts)
  Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
  __shared__ float
      smem[decltype(cosize(smem_layout))::value];  // (static-only allocation)
  Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
  Tensor smem_4x8_row =
      make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});

  // Register memory (static layouts only)
  Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
  Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});
  Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});
  Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);

  print(smem_4x8_col);
  print("\n");
  print(smem_4x8_row);
  print("\n");
  print(rmem_4x8_col);
  print("\n");
  print(rmem_4x8_row);
  print("\n");
  print(rmem_4x8_pad);
  print("\n");
  print(rmem_4x8_like);
  print("\n");
}

int main() {
  print("----------fundamemtal_operations----------\n");
  fundamemtal_operations();

  print("------------nonowning_tensors-------------\n");
  nonowning_tensors();

  print("-------------owning_tensors---------------\n");
  dim3 block(1);
  dim3 grid(1);
  owning_tensors<<<grid, block>>>();
}

void fundamemtal_operations() {
  float* A = new float[8]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

  Tensor ta =
      make_tensor(A, make_layout((Int<4>{}, Int<2>{}), (Int<1>{}, Int<4>{})));
  print("tensor ta: ");
  print_tensor(ta);  // tensor
  print("\n");

  auto shape = ta.shape();  // shape
  print("shape is: ");
  print(shape);
  print("\n");

  auto layout = ta.layout();  // layout
  print("layout is: ");
  print(layout);
  print("\n");

  auto stride = ta.stride();  // stride
  print("stride is: ");
  print(stride);
  print("\n");

  auto size = ta.size();  // size
  print("size is: ");
  print(size);
  print("\n");

  auto tile = ta.tile();  // tile
  print("tile is: ");
  print(tile);
  print("\n");

  delete[] A;
  A = nullptr;
}

void nonowning_tensors() {
  float* A = new float[8]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

  // Untagged pointers
  Tensor tensor_8 =
      make_tensor(A, make_layout(Int<8>{}));    // Construct with Layout
  Tensor tensor_8s = make_tensor(A, Int<8>{});  // Construct with Shape
  Tensor tensor_8d2 = make_tensor(A, 8, 2);  // Construct with Shape and Stride

  // Global memory (static or dynamic layouts)
  Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});
  Tensor gmem_8d = make_tensor(make_gmem_ptr(A), 8);
  Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
  Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape(8, Int<16>{}),
                                   make_stride(Int<16>{}, Int<1>{}));

  print(tensor_8);
  print("\n");
  print(tensor_8s);
  print("\n");
  print(tensor_8d2);
  print("\n");
  print(gmem_8s);
  print("\n");
  print(gmem_8d);
  print("\n");
  print(gmem_8sx16d);
  print("\n");
  print(gmem_8dx16s);
  print("\n");

  delete[] A;
  A = nullptr;
}
