#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <torch/nn/options/normalization.h>
using namespace at;
using namespace at::cuda::detail;
// using namespace caffe2;
template <typename scalar_t>
__device__ inline int indexOfMax(const scalar_t* ptr, const int size, const int stride) {
  scalar_t max_value = *ptr;
  int index =0;
  for(int i = 0; i < size; i++){
    scalar_t temp = *(ptr + i*stride);
    // printf("index: %f  \n", temp);
    if(temp >= max_value){
      max_value = temp;
      index = i;
    }
  }
  return index;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void optimal_path_kernel(
    const int nthreads,
    TensorInfo<scalar_t, int> input,
    TensorInfo<scalar_t, int> dp_buffer,
    TensorInfo<scalar_t, int> route_buffer,
    int M, int N, int H, int W,
    TensorInfo<scalar_t, int> output) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // printf("index: %d  \n", index);
    // index in N
    const int n = index % N;
    // index in M
    const int m = (index / N) % M;
    auto dp_buffer_ptr = dp_buffer.data + index*H*W;
    auto input_ptr = input.data + index*H*W;
    auto route_ptr = route_buffer.data + index*H*W*2;
    auto output_ptr = output.data + index*W;
    scalar_t max_before = 0;
    int idx = 0;

    for(int i=0; i < H; i++){
      for(int j=0; j < W; j++){
        if(i==0 && j==0){ 
          dp_buffer_ptr[i*W+j] = input_ptr[i*W+j];
        }
        else {
          if(i==0){
            max_before = dp_buffer_ptr[i*W+j-1];
            route_ptr[(i*W+j)*2] = i;
            route_ptr[(i*W+j)*2+1] = j-1;
          } else if(j==0) {
            max_before = 0;
            route_ptr[(i*W+j)*2] = i;
            route_ptr[(i*W+j)*2+1] = j;
          } else {
            idx = indexOfMax(dp_buffer_ptr+j-1, i+1, W);
            // idx=i;
            max_before = dp_buffer_ptr[idx*W+j-1];
            route_ptr[(i*W+j)*2] = idx;
            route_ptr[(i*W+j)*2+1] = j-1;
          }
          dp_buffer_ptr[i*W+j] = input_ptr[i*W+j] + max_before;
        }

      }//end of the 2-th for
    }//end of the 1-th for
    // scalar_t dp_sum = 0;
    // for(int i=0; i < H*W; i++){
    //   dp_sum += dp_buffer_ptr[i];
    // }
    // printf("sum: %d,%f  \n",index, dp_sum);
    idx = indexOfMax(dp_buffer_ptr+W-1,H,W);
    output_ptr[W-1] = idx;
    scalar_t i = route_ptr[(idx*W+W-1)*2];
    scalar_t j = route_ptr[(idx*W+W-1)*2 + 1];
    int k = W-1;
    while(k>0){
      output_ptr[static_cast<int>(j)]= i;
      i = route_ptr[static_cast<int>(i*W+j)*2];
      j = route_ptr[static_cast<int>(i*W+j)*2 + 1];
      k-=1;
    }
  }
}


 

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor optimal_path_cuda(const Tensor& grid) {
  auto M = grid.size(0);
  auto N = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto output = at::zeros({M, N, W}, grid.options());
  auto dp_buffer = at::zeros({M, N,H,W},grid.options());
  auto route_buffer = at::zeros({M, N,H,W,2},grid.options())-1;
  int count = static_cast<int>(N * M);
  // grid = grid.contiguous();
  // printf("M,N,H,W: %d,%d,%d,%d  \n", M,N,H,W);
  // float dp_sum = 0;
  // for(int i=0; i < M*N*H*W; i++){
  //   dp_sum += grid.data[i];
  // }
  // printf("grid sum: %f  \n", grid.sum());
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "optimal_path_cuda", [&] {
      optimal_path_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(dp_buffer),
          getTensorInfo<scalar_t, int>(route_buffer),
          static_cast<int>(M),
          static_cast<int>(N),
          static_cast<int>(H),
          static_cast<int>(W),
          getTensorInfo<scalar_t, int>(output));
    });
  }
  return output;
}