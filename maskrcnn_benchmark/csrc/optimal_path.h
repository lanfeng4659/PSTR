#pragma once
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include "cuda/vision.h"
#include <iostream>



at::Tensor OptimalPath_forward(
                                const at::Tensor& input){
    return optimal_path_cuda(input);
                                };
