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

void flash_attn_v2_cute_v1(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                           torch::Tensor& O);

void flash_attn_v2_cute_v2(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                           torch::Tensor& O, bool is_causal);