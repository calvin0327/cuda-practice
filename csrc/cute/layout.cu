#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <stdio.h>

int main() {
  auto shape = cute::make_shape(3, 2, 2);
  auto layout = cute::make_layout(shape);
  print(layout);
  return 0;
}
