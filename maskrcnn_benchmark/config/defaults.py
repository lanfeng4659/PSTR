# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RETRIEVAL_ONLY = False
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.FCOS_ON = False
_C.MODEL.POLYGON_DET = False
_C.MODEL.CHAR_ON = False
_C.MODEL.KE_ON = False
_C.MODEL.INST_ON = False
_C.MODEL.CHAR_INST_ON = False
_C.MODEL.MSR_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""
# _C.MODEL.BACK = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.AUGMENT = "PSSAugmentation"
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# The range of the smallest side for multi-scale training
_C.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)  # -1 means disabled and it will use MIN_SIZE_TRAIN
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.CROP_PROB_TRAIN = 0.0
_C.INPUT.CROP_SIZE_TRAIN = -1

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

_C.INPUT.FLIP_PROB_TRAIN = 0.0
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.TEXT = CN()
_C.DATASETS.TEXT.NUM_CHARS = 25
_C.DATASETS.TEXT.VOC_SIZE = 97

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8 #8
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 32
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# head options
# ---------------------------------------------------------------------------- #

_C.MODEL.ONE_STAGE_HEAD = 'align'


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.FREEZE_BN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_BN = False
_C.MODEL.FPN.USE_RELU = False
_C.MODEL.FPN.USE_DEFORMABLE = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
_C.MODEL.ROI_BOX_HEAD.USE_DFPOOL = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
_C.MODEL.ROI_BOX_HEAD.CLASS_WEIGHT = 1.0


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False
_C.MODEL.ROI_MASK_HEAD.USE_DFPOOL = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

_C.MODEL.ROI_INST_HEAD = CN()
_C.MODEL.ROI_INST_HEAD.PREDICTOR = "EmbeddingPredictor"

# ---------------------------------------------------------------------------- #
# One Stage Head Options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.BBOX_LOSS = CN()
_C.MODEL.HEAD.BBOX_LOSS.TYPE = 'IOULoss'
_C.MODEL.HEAD.BBOX_LOSS.ALPHA = 0.5
_C.MODEL.HEAD.BBOX_LOSS.GAMMA = 1.5
_C.MODEL.HEAD.BBOX_LOSS.BETA = 0.11
_C.MODEL.HEAD.BBOX_LOSS.WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
_C.MODEL.RESNETS.DEFORM_POOLING = False

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# _C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
_C.MODEL.RESNETS.STAGE_WITH_CONTEXT = (False, False, False, False)
_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
# avoid deep resnets with too many dcn layers
_C.MODEL.RESNETS.MAX_DCN_LAYER = 15
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_GN = True
_C.MODEL.FCOS.USE_BN = False
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_LIGHTWEIGHT = False  # use shufflev2 as head
_C.MODEL.FCOS.USE_DEFORMABLE = False
# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.USE_POLY = False
_C.MODEL.FCOS.USE_COUNT = False

# ---------------------------------------------------------------------------- #
# EAST Options
# ---------------------------------------------------------------------------- #
_C.MODEL.EAST = CN()
_C.MODEL.EAST.NUM_CLASSES = 2  # the number of classes including background
_C.MODEL.EAST.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.EAST.PRIOR_PROB = 0.01
_C.MODEL.EAST.INFERENCE_TH = 0.2
_C.MODEL.EAST.NMS_TH = 0.6
_C.MODEL.EAST.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_C.MODEL.EAST.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.EAST.LOSS_GAMMA = 2.0
_C.MODEL.EAST.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.EAST.USE_GN = True
_C.MODEL.EAST.USE_BN = False
_C.MODEL.EAST.USE_RELU = True
_C.MODEL.EAST.USE_LIGHTWEIGHT = False  # use shufflev2 as head
_C.MODEL.EAST.USE_DEFORMABLE = False
# the number of convolutions used in the cls and bbox tower
_C.MODEL.EAST.NUM_CONVS = 2
_C.MODEL.EAST.CENTER_SAMPLE = True
_C.MODEL.EAST.POS_RADIUS = 1.5
_C.MODEL.EAST.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# offset options for text recognition
# ---------------------------------------------------------------------------- #

_C.MODEL.OFFSET = CN()
_C.MODEL.OFFSET.PREDICTOR = 'polar'
_C.MODEL.OFFSET.KERNEL_SIZE = 3
_C.MODEL.OFFSET.STOP_OFFSETS = 1500


# ---------------------------------------------------------------------------- #
# neck options
# ---------------------------------------------------------------------------- #

_C.MODEL.NECK = CN()
_C.MODEL.NECK.CONV_BODY = 'none'
_C.MODEL.NECK.IN_CHANNELS = 256
_C.MODEL.NECK.NUM_LEVELS = 5
_C.MODEL.NECK.REFINE_LEVEL = 1
_C.MODEL.NECK.REFINE_TYPE = 'non_local'
_C.MODEL.NECK.USE_GN = False
_C.MODEL.NECK.USE_DEFORMABLE = False
_C.MODEL.NECK.LAST_STRIDE = 2

# ---------------------------------------------------------------------------- #
# align options
# ---------------------------------------------------------------------------- #

_C.MODEL.ALIGN = CN()
_C.MODEL.ALIGN.NUM_CONVS = 4
_C.MODEL.ALIGN.POOLER_RESOLUTION = (14, 64)
_C.MODEL.ALIGN.POOLER_CANONICAL_SCALE = 160
_C.MODEL.ALIGN.POOLER_SCALES = (0.125, 0.0625, 0.03125)
# ctc or attention
_C.MODEL.ALIGN.PREDICTOR = "ctc" 
_C.MODEL.ALIGN.PYRAMID_LAYERS = (2,3,4,5)
_C.MODEL.ALIGN.USE_ONLY_SPOTTER =False
_C.MODEL.ALIGN.USE_CTC_LOSS =False
_C.MODEL.ALIGN.USE_GLOBAL_LOCAL_SIMILARITY =False
_C.MODEL.ALIGN.USE_DYNAMIC_SIMILARITY = False
_C.MODEL.ALIGN.USE_PYRAMID = False
_C.MODEL.ALIGN.USE_NO_RNN = False
_C.MODEL.ALIGN.USE_DOMAIN_CLASSIFIER = False
_C.MODEL.ALIGN.USE_DOMAIN_ALIGN_LOSS = False
_C.MODEL.ALIGN.USE_WORD_INSTANCE_AUG = False
_C.MODEL.ALIGN.USE_FOCAL_L1_LOSS = False
_C.MODEL.ALIGN.USE_RES_LINK = False
_C.MODEL.ALIGN.USE_LOOK_UP = False
_C.MODEL.ALIGN.USE_CHARACTER_AWARENESS = False
_C.MODEL.ALIGN.IS_CHINESE = False
_C.MODEL.ALIGN.USE_ALONG_LOSS = False
_C.MODEL.ALIGN.USE_N_GRAM_ED = False
_C.MODEL.ALIGN.USE_COMMON_SPACE = False
_C.MODEL.ALIGN.USE_HANMING = False
_C.MODEL.ALIGN.USE_STEP = False
_C.MODEL.ALIGN.USE_CHAR_COUNT = False
_C.MODEL.ALIGN.USE_LEN_EMBED = False
_C.MODEL.ALIGN.USE_CHAR_COUNT_DETACH = False
_C.MODEL.ALIGN.USE_CONTRASTIVE_LOSS = False
_C.MODEL.ALIGN.USE_BOX_AUG = False
_C.MODEL.ALIGN.USE_TEXTNESS = False
_C.MODEL.ALIGN.USE_RETRIEVAL = True
_C.MODEL.ALIGN.USE_IOU_PREDICTOR = False
_C.MODEL.ALIGN.USE_WORD_AUG = False
_C.MODEL.ALIGN.USE_SCALENET = True
_C.MODEL.ALIGN.USE_MIL= True
_C.MODEL.ALIGN.USE_PARTIAL_SAMPLES = False
_C.MODEL.ALIGN.USE_CONTRASTIVE_LOSS = False
_C.MODEL.ALIGN.DET_SCORE = (0.2, 0.2, 0.05)

_C.MODEL.ALIGN.WORDEMBEDDING = CN()
_C.MODEL.ALIGN.WORDEMBEDDING.USE_PADDING = False
_C.MODEL.ALIGN.WORDEMBEDDING.USE_TEXT_FEAT = False
_C.MODEL.ALIGN.WORDEMBEDDING.USE_CUT_WORD = False
_C.MODEL.ALIGN.WORDEMBEDDING.USE_SEGMENT_EMBEDDING = False
_C.MODEL.ALIGN.WORDEMBEDDING.EMBEDDING_GROUP = 1
# ---------------------------------------------------------------------------- #
# attebtion options
# ---------------------------------------------------------------------------- #

_C.MODEL.ATTENTION = CN()
_C.MODEL.ATTENTION.NUM_CONVS = 4
_C.MODEL.ATTENTION.POOLER_RESOLUTION = (14, 64)
_C.MODEL.ATTENTION.POOLER_CANONICAL_SCALE = 160
_C.MODEL.ATTENTION.POOLER_SCALES = (0.125, 0.0625, 0.03125)
# ctc or attention
_C.MODEL.ATTENTION.PREDICTOR = "ctc"
_C.MODEL.ATTENTION.IS_CHINESE = False
_C.MODEL.ATTENTION.USE_BOX_AUG = False
_C.MODEL.ATTENTION.USE_RETRIEVAL = True
_C.MODEL.ATTENTION.USE_WORD_AUG = False
# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = False

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.SCHEDULER = 'multistep'

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.ONE_STAGE_HEAD_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

# multistep
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
# poly
_C.SOLVER.POLY_POWER = 0.9

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
_C.TEST.BBOX_AUG.SCALE_H_FLIP = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.IS_LOAD_OPTIMIZER = True
_C.IS_LOAD_SCHEDULER = True
_C.PROCESS = CN()
_C.PROCESS.PNMS = False
_C.PROCESS.NMS_THRESH = 0.4

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
_C.SYNCBN = False

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

# ---------------------------------------------------------------------------- #
# DARTS Options
# ---------------------------------------------------------------------------- #
_C.DARTS = CN()
_C.DARTS.LR_A = 0.001
_C.DARTS.WD_A = 0.001
_C.DARTS.T_MAX = 2500  # cosine lr time
_C.DARTS.LR_END = 0.0001
_C.DARTS.ARCH_START_ITER = 5000
_C.DARTS.TIE_CELL = False
