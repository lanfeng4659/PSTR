# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign, BezierAlign, ROIAlignAdaptive
from maskrcnn_benchmark.layers import ModulatedDeformRoIPoolingPack
from maskrcnn_benchmark.layers.arbitrary_roi_align import ArbitraryROIAlign
from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

    def get_random(self, level):
        """ Generate a random roi for target level
        """
        xmin, ymin, xmax, ymax = torch.tensor


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio,
                 output_channel=256, canonical_scale=160,
                 mode='align'):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            if mode == 'align':
                pooler = ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
            elif mode == 'alignadaptive':
                pooler = ROIAlignAdaptive(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
            # elif mode == 'deformable':
            #     pooler = ModulatedDeformRoIPoolingPack(
            #         spatial_scale=scale, out_size=output_size[0],
            #         out_channels=output_channel, no_trans=False,
            #         group_size=1, trans_std=0.1)
            # elif mode == 'bezier':
            #     pooler = BezierAlign(
            #         output_size, spatial_scale=scale, sampling_ratio=1)
            else:
                raise NotImplementedError()
            poolers.append(pooler)
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_scale=canonical_scale)
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        if isinstance(boxes[0], torch.Tensor):
            concat_boxes = cat([b for b in boxes], dim=0)
        else:
            concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes).to(x[0].device)
        # print(rois)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, *self.output_size),
            dtype=dtype,
            device=device,
        )
        # for f in x:
        #     print(f.shape)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            # print(rois_per_level)
            # print(level, per_level_feature.shape)
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)

        return result

class PolyPooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, num_points, output_size, scales, sampling_ratio,
                 output_channel=256, canonical_scale=160,
                 mode='align'):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PolyPooler, self).__init__()
        poolers = []
        self.num_points = num_points
        # print(scales)
        # exit()
        self.scales = scales
        for scale in scales:
            pooler = ArbitraryROIAlign(output_size, self.num_points*2, [0,0])
            poolers.append(pooler)
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        self.idx = list(range(self.num_points)) + list(range(self.num_points, self.num_points*2))[::-1]
        # exit()
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_scale=canonical_scale)
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        if isinstance(boxes[0], torch.Tensor):
            # print('box')
            concat_boxes = cat([b for b in boxes], dim=0)
            device, dtype = concat_boxes.device, concat_boxes.dtype
            ids = cat(
                [
                    torch.full((len(b), 1), i, dtype=dtype, device=device)
                    for i, b in enumerate(boxes)
                ],
                dim=0,
            )
        else:
            # print("poly")
            concat_boxes = cat([b.get_field("polys") for b in boxes], dim=0)
            device, dtype = concat_boxes.device, concat_boxes.dtype
            ids = cat(
                [
                    torch.full((len(b.get_field("polys")), 1), i, dtype=dtype, device=device)
                    for i, b in enumerate(boxes)
                ],
                dim=0,
            )
        rois = torch.cat([ids, concat_boxes], dim=1)
        # print(rois)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes).to(x[0].device)
        # print(rois)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)
        # import ipdb; ipdb.set_trace()
        num_rois = len(rois)
        num_channels = x[0].shape[1]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, *self.output_size),
            dtype=dtype,
            device=device,
        )
        h, w = x[0].shape[-2:]
        wh=torch.FloatTensor([w,h])[None,None,:].to(device)/self.scales[0]
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            polys_in_this_level = rois_per_level[:,1:].reshape([-1,self.num_points*2,2])/wh
            index_in_which_img = rois_per_level[:,0].cpu()
            result[idx_in_level] = pooler(per_level_feature, polys_in_this_level[:,self.idx], index_in_which_img).to(dtype)
        return result
class PolyPoolerTextLenSensitive(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, num_points, output_size_list, lens_area, scales, sampling_ratio,
                 output_channel=256, canonical_scale=160,
                 mode='align'):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PolyPoolerTextLenSensitive, self).__init__()
        poolers = []
        self.num_points = num_points
        assert len(lens_area) == len(output_size_list)
        self.lens_area = lens_area
        self.scales = scales
        for output_size in output_size_list:
            # print(output_size)
            pooler = ArbitraryROIAlign(output_size, self.num_points*2, [0,0])
            poolers.append(pooler)
        self.poolers = nn.ModuleList(poolers)
        self.output_size_list = output_size_list
        self.idx = list(range(self.num_points)) + list(range(self.num_points, self.num_points*2))[::-1]
        # exit()
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_scale=canonical_scale)
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        
        concat_boxes = cat([b.get_field("polys") for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b.get_field("polys")), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois
    def regroup_proposal_by_len(self, lens):
        def map_pooler_id(len_):
            for idx, len_max in enumerate(self.lens_area):
                if len_ <= len_max:
                    return idx
        # print(lens)
        pooler_id = torch.tensor([map_pooler_id(l) for l in lens])
        # print(pooler_id)
        return pooler_id
    # def forward(self, x, boxes, texts, is_training=True):
    #     if is_training:
    #         self.forward_train(x, boxes, texts)
    #     else:
    #         self.forward_test(x, boxes, texts)

    def forward(self, x, boxes, texts):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.scales)
        lens = [len(w) for w in texts]
        pooler_ids = self.regroup_proposal_by_len(lens).to(x[0].device)
        rois = self.convert_to_roi_format(boxes).to(x[0].device)


        levels = self.map_levels(boxes)

        # num_rois = len(rois)
        num_channels = x[0].shape[1]

        dtype, device = x[0].dtype, x[0].device
        results = [[] for i in range(len(self.output_size_list))]
        texts_batch = [[] for i in range(len(self.output_size_list))]
        h, w = x[0].shape[-2:]
        wh=torch.FloatTensor([w,h])[None,None,:].to(device)/self.scales[0]
        for pooler_id, (pooler, result, tb) in enumerate(zip(self.poolers, results,texts_batch)):
            # print(pooler.output_size)
            for level, per_level_feature in enumerate(x):
                idx_in_pooler_level = torch.nonzero((pooler_ids==pooler_id) & (levels == level)).squeeze(1)
                if idx_in_pooler_level.numel()==0:
                    continue
                rois_per_level = rois[idx_in_pooler_level]
                polys_in_this_level = rois_per_level[:,1:].reshape([-1,self.num_points*2,2])/wh
                index_in_which_img = rois_per_level[:,0].cpu()
                result.append(
                    pooler(per_level_feature, polys_in_this_level[:,self.idx], index_in_which_img).to(dtype)
                ) 
                tb.extend([texts[v] for v in idx_in_pooler_level])
        return [torch.cat(result) for result in results if len(result) > 0], [t for t in texts_batch if len(t)>0]
def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
