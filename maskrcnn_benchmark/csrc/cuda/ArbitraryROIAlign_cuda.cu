#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

enum class ArbitraryROIAlignInterpolation {Bilinear, Nearest};
enum class ArbitraryROIAlignPadding {Zeros, Border, Reflection};
// namespace at { namespace native {
using namespace at;
using namespace at::cuda::detail;
  static __forceinline__ __device__
  float clip_coordinates(float in, int clip_limit) {
    return ::min(static_cast<float>(clip_limit - 1), ::max(in, 0.f));
  }

  // clip_coordinates_set_grad works similarly to clip_coordinates except that
  // it also returns the `d output / d input` via pointer argument `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template <typename scalar_t>
  static __forceinline__ __device__
  float clip_coordinates_set_grad(float in, int clip_limit, scalar_t *grad_in) {
    if (in < 0.f) {
      *grad_in = static_cast<scalar_t>(0);
      return 0.f;
    } else {
      float max = static_cast<float>(clip_limit - 1);
      if (in > max) {
        *grad_in = static_cast<scalar_t>(0);
        return max;
      } else {
        *grad_in = static_cast<scalar_t>(1);
        return in;
      }
    }
  }

  static __forceinline__ __device__
  float reflect_coordinates(float in, int clip_limit) {
    if (clip_limit == static_cast<int>(1)) {
      return 0.f;
    }
    in = ::fabs(in);
    float max = static_cast<float>(clip_limit - 1);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    float extra = ::fmod(in, max);
    int flips = static_cast<int>(::floor(in / max));
    if (flips % 2 == 0) {
      return extra;
    } else {
      return max - extra;
    }
  }

  // reflect_coordinates_set_grad works similarly to reflect_coordinates except
  // that it also returns the `d output / d input` via pointer argument
  // `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template <typename scalar_t>
  static __forceinline__ __device__
  float reflect_coordinates_set_grad(float in, int clip_limit, scalar_t *grad_in) {
    if (clip_limit == static_cast<int>(1)) {
      *grad_in = static_cast<scalar_t>(0);
      return 0.f;
    }
    int grad_in_mult_;
    if (in < 0.f) {
      grad_in_mult_ = -1;
      in = -in;
    } else {
      grad_in_mult_ = 1;
    }
    float max = static_cast<float>(clip_limit - 1);
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    float extra = ::fmod(in, max);
    int flips = static_cast<int>(::floor(in / max));
    if (flips % 2 == 0) {
      *grad_in = static_cast<scalar_t>(grad_in_mult_);
      return extra;
    } else {
      *grad_in = static_cast<scalar_t>(-grad_in_mult_);
      return max - extra;
    }
  }

  static __forceinline__ __device__
  bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }


  template<typename scalar_t>
  static __forceinline__ __device__
  void safe_add_2d(scalar_t *data, int h, int w,
                   int sH, int sW, int H, int W,
                   scalar_t delta) {
    if (within_bounds_2d(h, w, H, W)) {
      atomicAdd(data + h * sH + w * sW, delta);
    }
  }


  template <typename scalar_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void arbitrary_roi_align_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> indexs,
      TensorInfo<scalar_t, int> output,
      const ArbitraryROIAlignInterpolation interpolation_mode,
      const ArbitraryROIAlignPadding padding_mode) {
    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
    int out_H = grid.sizes[1];
    int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int out_sN = output.strides[0];
    int out_sC = output.strides[1];
    int out_sH = output.strides[2];
    int out_sW = output.strides[3];
    CUDA_KERNEL_LOOP(index, nthreads) {
      // index in width
      const int w = index % out_W;
      // index in height
      const int h = (index / out_W) % out_H;
      // index of grids
      const int n = index / (out_H * out_W);
      // index of input
      const int idx = indexs.data[n];
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
      // get the corresponding input x, y co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      if (padding_mode == ArbitraryROIAlignPadding::Border) {
        // clip coordinates to image borders
        ixf = clip_coordinates(ixf, inp_W);
        iyf = clip_coordinates(iyf, inp_H);
      } else if (padding_mode == ArbitraryROIAlignPadding::Reflection) {
        // reflect coordinates by image borders
        ixf = reflect_coordinates(ixf, inp_W);
        iyf = reflect_coordinates(iyf, inp_H);
      }

      ix = static_cast<scalar_t>(ixf);
      iy = static_cast<scalar_t>(iyf);
      if (interpolation_mode == ArbitraryROIAlignInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ixf));
        int iy_nw = static_cast<int>(::floor(iyf));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input.data + idx * inp_sN;
        // int64_t index_data = n * out_sN + h * out_sH + w * out_sW;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
          // output_data[index_data] = static_cast<scalar_t>(0);
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
        }
      } else if (interpolation_mode == ArbitraryROIAlignInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ixf));
        int iy_nearest = static_cast<int>(::round(iyf));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + idx * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }


  template <typename scalar_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void arbitrary_roi_align_backward_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> grad_output,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> indexs,
      TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
      const ArbitraryROIAlignInterpolation interpolation_mode,
      const ArbitraryROIAlignPadding padding_mode) {

    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
    int out_H = grid.sizes[1];
    int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sH = grad_output.strides[2];
    int gOut_sW = grad_output.strides[3];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sH = grad_input.strides[2];
    int gInp_sW = grad_input.strides[3];
    int gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      const int idx = indexs.data[n];
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];

      // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
      float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

      // multipliers for gradients on ix and iy
      // E.g.,  0 for out-of-bound indices when ArbitraryROIAlignPadding::Border
      scalar_t gix_mult, giy_mult;
      if (padding_mode == ArbitraryROIAlignPadding::Border) {
        // clip coordinates to image borders
        ixf = clip_coordinates_set_grad(ixf, inp_W, &gix_mult);
        iyf = clip_coordinates_set_grad(iyf, inp_H, &giy_mult);
      } else if (padding_mode == ArbitraryROIAlignPadding::Reflection) {
        // reflect coordinates by image borders
        ixf = reflect_coordinates_set_grad(ixf, inp_W, &gix_mult);
        iyf = reflect_coordinates_set_grad(iyf, inp_H, &giy_mult);
      } else {  // padding_mode == ArbitraryROIAlignPadding::Zeros
        gix_mult = static_cast<scalar_t>(1);
        giy_mult = static_cast<scalar_t>(1);
      }

      if (interpolation_mode == ArbitraryROIAlignInterpolation::Bilinear) {
        ix = static_cast<scalar_t>(ixf);
        iy = static_cast<scalar_t>(iyf);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ixf));
        int iy_nw = static_cast<int>(::floor(iyf));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + idx * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + idx * inp_sN;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          // calculate and set grad_input
          safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
          safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
          safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
          safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

          // calculate grad_grid
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            gix -= nw_val * (iy_se - iy) * gOut;
            giy -= nw_val * (ix_se - ix) * gOut;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            gix += ne_val * (iy_sw - iy) * gOut;
            giy -= ne_val * (ix - ix_sw) * gOut;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            gix -= sw_val * (iy - iy_ne) * gOut;
            giy += sw_val * (ix_ne - ix) * gOut;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            gix += se_val * (iy - iy_nw) * gOut;
            giy += se_val * (ix - ix_nw) * gOut;
          }
        }

        // un-normalize grad_grid values back to [-1, 1] constraints
        gix = gix * (inp_W - 1.f) / 2;
        giy = giy * (inp_H - 1.f) / 2;

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } else if (interpolation_mode == ArbitraryROIAlignInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ixf));
        int iy_nearest = static_cast<int>(::round(iyf));

        // assign nearest neighor pixel value to output pixel
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + idx * gInp_sN;
        for (int c = 0; c < C; ++c, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          // calculate and set grad_input
          safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW);
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      }
    }
  }


// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor arbitrary_roi_align_forward_cuda(const Tensor& input, const Tensor& grid, const Tensor& indexs,
                            int64_t interpolation_mode, int64_t padding_mode) {
  auto N = grid.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::empty({N, input.size(1), H, W}, input.options());
  int count = static_cast<int>(N * H * W);
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "arbitrary_roi_align_forward_cuda", [&] {
      arbitrary_roi_align_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(indexs),
          getTensorInfo<scalar_t, int>(output),
          static_cast<ArbitraryROIAlignInterpolation>(interpolation_mode),
          static_cast<ArbitraryROIAlignPadding>(padding_mode));
    });
  }
  return output;
}


// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
arbitrary_roi_align_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& grid, const Tensor& indexs,
                              int64_t interpolation_mode, int64_t padding_mode) {
  auto N = grid.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);
  int count = static_cast<int>(N * H * W);
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "arbitrary_roi_align_backward_cuda", [&] {
      arbitrary_roi_align_backward_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(grad_output),
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(indexs),
          getTensorInfo<scalar_t, int>(grad_input),
          getTensorInfo<scalar_t, int>(grad_grid),
          static_cast<ArbitraryROIAlignInterpolation>(interpolation_mode),
          static_cast<ArbitraryROIAlignPadding>(padding_mode));
    });
  }
  return std::make_tuple(grad_input, grad_grid);
}