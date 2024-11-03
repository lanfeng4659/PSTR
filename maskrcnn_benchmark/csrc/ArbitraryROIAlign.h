#pragma once
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include "cuda/vision.h"
#include <iostream>



at::Tensor ArbitraryROIAlign_forward(
                                const at::Tensor& input, 
                                const at::Tensor& grid,
                                const at::Tensor& indexs,
                                int64_t interpolation_mode, 
                                int64_t padding_mode){
    return arbitrary_roi_align_forward_cuda(
                                input, 
                                grid, 
                                indexs, 
                                interpolation_mode, 
                                padding_mode);
                                };

std::tuple<at::Tensor, at::Tensor>
    ArbitraryROIAlign_backward(
                                const at::Tensor& grad_output, 
                                const at::Tensor& input, 
                                const at::Tensor& grid, 
                                const at::Tensor& indexs,
                                int64_t interpolation_mode, 
                                int64_t padding_mode){
    return arbitrary_roi_align_backward_cuda(
                                grad_output, 
                                input, 
                                grid, 
                                indexs,
                                interpolation_mode, 
                                padding_mode);
                                }
// Interface for Python
