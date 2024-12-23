import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

# from maskrcnn_benchmark.data.datasets.bezier import BEZIER

class Selector(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            fpn_strides=None,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(Selector, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh #0.05
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness, image_sizes, offsets=None):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        # poly_regression = poly_regression.view(N, 8, H, W).permute(0, 2, 3, 1)
        # poly_regression = poly_regression.reshape(N, -1, 8)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        if offsets is not None:
            offsets = torch.cat((offsets, mask), dim=1)
            offsets = offsets.permute(0, 2, 3, 1).reshape(N, H * W, -1)
        self.pre_nms_thresh = 0.01
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            # per_poly_regression = poly_regression[i]
            # per_poly_regression = per_poly_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if offsets is not None:
                per_offsets = offsets[i]
                per_offsets = per_offsets[per_box_loc]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                # per_poly_regression = per_poly_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if offsets is not None:
                    per_offsets = per_offsets[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            # bezier_detections = per_locations[:, [1, 0]].unsqueeze(1) + per_bezier_regression.view(-1, 8, 2) 
            # bezier_detections = bezier_detections.view(-1, 16)
            # print(per_locations.shape)
            # poly_detections = per_locations.unsqueeze(1) + per_poly_regression.view(-1, 4, 2) 


            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class.float())
            boxlist.add_field("scores", per_box_cls)
            # boxlist.add_field("polys", poly_detections)
            if offsets is not None:
                boxlist.add_field("offsets", per_offsets[:, :max_len * 2])
                boxlist.add_field("rec_masks", per_offsets[:, max_len * 2:].sigmoid())
                boxlist.add_field("locations", per_locations)

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(
            self, locations, box_cls, box_regression,
            centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            poly_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, c) in enumerate(zip(
                locations, box_cls, box_regression, centerness)):
            """
            if len(f) == 0:
                f = None
            else:
                f = f * self.fpn_strides[i]
            """
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        has_offsets = boxlists[0].has_field("offsets")
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            if has_offsets:
                offsets = boxlists[i].get_field("offsets")
                locations = boxlists[i].get_field("locations")
                rec_masks = boxlists[i].get_field("rec_masks")
            # polys = boxlists[i].get_field("polys")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                # polys_j = polys[inds, :].view(-1, 8)

                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                # boxlist_for_class.add_field("polys", polys_j)

                if has_offsets:
                    boxlist_for_class.add_field(
                        "offsets", offsets[inds])
                    boxlist_for_class.add_field(
                        "locations", locations[inds])
                    boxlist_for_class.add_field(
                        "rec_masks", rec_masks[inds])

                # boxlist_for_class = boxlist_nms(
                #     boxlist_for_class, self.nms_thresh,
                #     score_field="scores"
                # )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.float,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)
            # print(number_of_detections)
            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

def build_selector(config, is_train=True):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    if is_train:
        fpn_post_nms_top_n = 100
        # fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
        pre_nms_thresh = 0.01

    box_selector = Selector(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        fpn_strides=config.MODEL.FCOS.FPN_STRIDES,
    )

    return box_selector
