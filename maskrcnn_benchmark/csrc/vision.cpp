// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "BezierAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "deform_conv.h"
#include "deform_pool.h"
#include "ArbitraryROIAlign.h"
#include "optimal_path.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("optimal_path", &OptimalPath_forward, "optimal path");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("bezier_align_forward", &BezierAlign_forward, "BezierAlign_forward");
  m.def("bezier_align_backward", &BezierAlign_backward, "BezierAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  // dcn-v2
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "deform_psroi_pooling_forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "deform_psroi_pooling_backward");


  m.def("ArbitraryROIAlign_forward", &ArbitraryROIAlign_forward, "ArbitraryROIAlign_forward");
  m.def("ArbitraryROIAlign_backward", &ArbitraryROIAlign_backward, "ArbitraryROIAlign_backward");
}
