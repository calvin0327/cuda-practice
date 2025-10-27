#pragma once

#include <torch/all.h>
#include <torch/library.h>

void relu_f32(torch::Tensor& x, torch::Tensor& y);
void relu_f32x4(torch::Tensor& x, torch::Tensor& y);

void elu_f32(torch::Tensor& x, torch::Tensor& y);
void elu_f16(torch::Tensor& x, torch::Tensor& y);

void sgemm_naive_f32(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
void sgemm_shared_f32(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
void sgemm_t_8x8_shared_f32x4(torch::Tensor& a, torch::Tensor& b,
                              torch::Tensor& c);