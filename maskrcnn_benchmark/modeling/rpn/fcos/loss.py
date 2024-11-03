"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.utils.comm import reduce_sum, get_world_size
from maskrcnn_benchmark.layers import smooth_l1_loss


INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.loc_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.object_sizes_of_interest = soi
        self.count_loss = nn.CrossEntropyLoss()

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            # print(level, n_p)
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        # print(gt_xs.shape, gt_ys.shape,num_gts, inside_gt_bbox_mask.shape)
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = self.object_sizes_of_interest
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        # labels, reg_targets, bezier_targets = self.compute_targets_for_locations(
        #     points_all_level, targets, expanded_object_sizes_of_interest
        # )
        labels, reg_targets, poly_targets,count_targets, is_in_bboxes = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            if self.use_poly:
                poly_targets[i] = torch.split(poly_targets[i], num_points_per_level, dim=0)
            if self.use_count:
                count_targets[i] = torch.split(count_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        poly_targets_level_first = []
        count_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            # normalize regression targets
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level]
                           for reg_targets_per_im in reg_targets],
                          dim=0) / self.strides[level]
            )
            if self.use_poly:
                poly_targets_level_first.append(
                    torch.cat([poly_targets_per_im[level]
                            for poly_targets_per_im in poly_targets],
                            dim=0) / self.strides[level]
                )
            if self.use_count:
                count_targets_level_first.append(
                torch.cat([counts_per_im[level] for counts_per_im in count_targets], dim=0)
            )
        return labels_level_first, reg_targets_level_first, poly_targets_level_first,count_targets_level_first, is_in_bboxes#bezier_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        poly_targets = []
        count_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()
            # print("area:",area.shape)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            

            # polys = targets_per_im.get_field("polys").view(-1, 4, 2).type_as(ys)
            if self.use_poly:
                polys = targets_per_im.get_field("polys").view(-1, 14, 2).type_as(ys)
                # assert polys.size(0) == bboxes.size(0)
                x_targets = polys[:, :, 0][None] - xs[:, None, None]
                y_targets = polys[:, :, 1][None] - ys[:, None, None]
                poly_targets_per_im = torch.stack((x_targets, y_targets), dim=3)
                # poly_targets_per_im = poly_targets_per_im.view(xs.size(0), bboxes.size(0), 8)
                poly_targets_per_im = poly_targets_per_im.view(xs.size(0), bboxes.size(0), 28)
            # print(reg_targets_per_im.shape,poly_targets_per_im.shape, bboxes.shape, polys.shape)
            # bezier points are relative distances from center to control points
            # bezier_pts = targets_per_im.get_field("beziers").bbox.view(-1, 8, 2)
            # bezier_pts = torch.zeros([bboxes.size(0),8,2]).type_as(ys)
            # y_targets = bezier_pts[:, :, 0][None] - ys[:, None, None]
            # x_targets = bezier_pts[:, :, 1][None] - xs[:, None, None]
            # bezier_targets_per_im = torch.stack((y_targets, x_targets), dim=3)
            # bezier_targets_per_im = bezier_targets_per_im.view(xs.size(0), bboxes.size(0), 16)

            if self.center_sample:
                # print("hello")
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_points_per_level,
                    xs, ys, radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            # print(locations.shape)
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            if self.use_poly:
                # print(poly_targets_per_im.shape, reg_targets_per_im.shape, bboxes.shape)
                poly_targets_per_im = poly_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            # print(labels_per_im.shape)
            labels_per_im[locations_to_min_area == INF] = 0
            # print(labels_per_im)
            if self.use_count:
                counts_per_im = torch.tensor([min(len(text), 19) for text in targets_per_im.get_field("texts")]).type_as(labels_per_im)
                counts_per_im = counts_per_im[locations_to_gt_inds]
                count_targets.append(counts_per_im)


            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            if self.use_poly:
                poly_targets.append(poly_targets_per_im)
        if not self.use_poly:
            poly_targets = None
        return labels, reg_targets, poly_targets,count_targets, labels[0]#bezier_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression,poly_regression, centerness, count_pred, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        self.use_poly = poly_regression is not None
        self.use_count = count_pred is not None
        num_classes = box_cls[0].size(1)
        if self.use_count:
            num_count = count_pred[0].size(1)
        # labels, reg_targets, bezier_targets = self.prepare_targets(locations, targets)
        # labels, reg_targets,is_in_bboxes = self.prepare_targets(locations, targets)
        labels, reg_targets, poly_targets, count_targets, is_in_bboxes = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        poly_regression_flatten=[]

        centerness_flatten = []
        labels_flatten = []

        reg_targets_flatten = []
        poly_targets_flatten = []

        count_preds_flatten = []
        count_targets_flatten = []

        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            if self.use_poly:
                poly_regression_flatten.append(poly_regression[l].permute(0, 2, 3, 1).reshape(-1, 28))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            if self.use_poly:
                poly_targets_flatten.append(poly_targets[l].reshape(-1, 28))
            centerness_flatten.append(centerness[l].reshape(-1))

            if self.use_count:
                count_preds_flatten.append(count_pred[l].permute(0, 2, 3, 1).reshape(-1, num_count))
                count_targets_flatten.append(count_targets[l].reshape(-1))


        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        if self.use_poly:
            poly_regression_flatten = torch.cat(poly_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)

        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        if self.use_poly:
            poly_targets_flatten = torch.cat(poly_targets_flatten, dim=0)
        if self.use_count:
            count_preds_flatten = torch.cat(count_preds_flatten, dim=0)
            count_targets_flatten = torch.cat(count_targets_flatten, dim=0)
        
        # -1 difficult 
        # 0 bg 
        # 1 text
        not_difficult_inds = torch.nonzero(labels_flatten > -1).squeeze(1) 
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1) 
        num_pos_per_gpu = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_per_gpu])).item()

        box_regression_flatten = box_regression_flatten[pos_inds]
        if self.use_poly:
            poly_regression_flatten = poly_regression_flatten[pos_inds]

        reg_targets_flatten = reg_targets_flatten[pos_inds]
        if self.use_poly:
            poly_targets_flatten = poly_targets_flatten[pos_inds]

        if self.use_count:
            count_preds_flatten = count_preds_flatten[pos_inds]
            count_targets_flatten = count_targets_flatten[pos_inds]


        centerness_flatten = centerness_flatten[pos_inds]
        # print(box_cls_flatten.shape)
        box_cls_flatten = box_cls_flatten[not_difficult_inds]
        # print(box_cls_flatten.shape)
        labels_flatten = labels_flatten[not_difficult_inds]
        # print(labels_flatten, box_cls_flatten,total_num_pos)
        # print(box_cls_flatten.max(),labels_flatten.min())
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / max(total_num_pos / num_gpus, 1.0)  # add N to avoid dividing by a zero

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            sum_centerness_targets = centerness_targets.sum()
            sum_centerness_targets = reduce_sum(sum_centerness_targets).item()
            # print(box_regression_flatten.shape)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / (sum_centerness_targets / num_gpus)
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / max(total_num_pos / num_gpus, 1.0)
            if self.use_count:
                # print(count_targets_flatten)
                count_loss = self.count_loss(
                count_preds_flatten,
                count_targets_flatten.long()
                ) / (sum_centerness_targets / num_gpus)
                count_loss = count_loss*10
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
            if self.use_count:
                count_loss = count_preds_flatten.sum()*0
        if not self.use_count:
            count_loss = None
        if self.use_poly:
            poly_loss = F.smooth_l1_loss(
                poly_regression_flatten, poly_targets_flatten, reduction="none")
            poly_loss = ((poly_loss.mean(dim=-1) * centerness_targets).sum()
                            / (sum_centerness_targets / num_gpus))
            return cls_loss, reg_loss, poly_loss, centerness_loss,count_loss,is_in_bboxes
        else:
            return cls_loss, reg_loss, None, centerness_loss,count_loss,is_in_bboxes

    def compute_offsets_targets(self, mask_targets, reg_targets):
        num_chars = mask_targets.sum(dim=1).long()
        N, K = mask_targets.size()
        offsets_x = torch.zeros(N, K, dtype=torch.float32, device=mask_targets.device)
        offsets_y = torch.zeros(N, K, dtype=torch.float32, device=mask_targets.device)
        for i, (nc, reg) in enumerate(zip(num_chars, reg_targets)):
            xs = (reg[2] + reg[0]) * (torch.tensor(list(range(nc)),
                                                   dtype=torch.float32,
                                                   device=mask_targets.device) * 2 + 1) / (nc * 2) - reg[0]
            offsets_x[i, :nc] = xs
            offsets_y[i, :nc] = (reg[3] - reg[1]) / 2
        return torch.stack((offsets_y, offsets_x), dim=2).view(N, -1)


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
