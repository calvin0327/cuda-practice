#pragma once

#include <torch/all.h>
#include <torch/library.h>

void sgemm_naive_f32(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
