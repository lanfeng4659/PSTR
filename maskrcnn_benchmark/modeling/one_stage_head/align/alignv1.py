import torch
from torch import nn
from torch.nn import functional as F
import cv2
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler, PolyPooler, PolyPoolerTextLenSensitive
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos
from maskrcnn_benchmark.utils.text_util import TextGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, cat_boxlist_texts
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.chinese_utils import load_chars
from maskrcnn_benchmark.modeling.one_stage_head.align.tfhead import TextFeat
from maskrcnn_benchmark.modeling.one_stage_head.align.matric_similarity_learning import matric_similarity_learning_loss
from maskrcnn_benchmark.modeling.one_stage_head.align.attention import AttentionRecognitionHead
# from maskrcnn_benchmark.layers.arbitrary_roi_align import ArbitraryROIAlign
from torch.autograd import Variable
import string
import random
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers import SigmoidFocalLoss
# from .box_aug import make_box_aug
from fonts import load_fonts
from collections import Iterable
INF = 10000000000
class ScaleNet(nn.Module):
    def __init__(self,in_channels, size=(4,15)):
        super(ScaleNet, self).__init__()
        conv_func = conv_with_kaiming_uniform(True, True, use_deformable=False, use_bn=False)
        self.h, self.w = size
        # self.use_res_link = False
        # # self.rescale = nn.Upsample(size=(self.h, self.w), mode='bilinear', align_corners=False)
        # if self.use_res_link:
        #     self.res_link = nn.Sequential(
        #         conv_func(in_channels, in_channels, 3, stride=(2, 1)),
        #         conv_func(in_channels, in_channels, 3, stride=(2, 1))
        #     )
        self.attn_conv = nn.Sequential(
            conv_func(in_channels, in_channels, 3, stride=(2, 1)),
            conv_func(in_channels, in_channels//2, 3, stride=(2, 1))
        )
        self.attn = nn.Sequential(
            nn.Linear(self.w*in_channels//2,512),
            nn.ReLU(),
            nn.Linear(512,self.w*self.w),
        )
        self.f_conv = nn.Sequential(
            conv_func(in_channels, in_channels, 3, stride=(2, 1)),
            conv_func(in_channels, in_channels, 3, stride=(2, 1))
        )
        # self.attention = conv_func(in_channels, len(neighbors), (len(neighbors),3), stride=(len(neighbors), 1),padding=(0, dilation * (3 - 1) // 2))
    def forward(self, x):
        b = x.size(0)
        af = self.attn_conv(x).view((b,-1))
        att = self.attn(af).view((b,self.w, self.w)).softmax(dim=-1)
        # ff = torch.bmm(self.f_conv(self.rescale(x)).squeeze(dim=-2),att)
        ff = torch.bmm(self.f_conv(x).squeeze(dim=-2),att)
        # if self.use_res_link:
        #     ff = ff + self.res_link(x).squeeze(dim=-2)

        if self.training:
            return ff.unsqueeze(dim=-2)
        else:
            return ff.unsqueeze(dim=-2), att

class CTCPredictor(nn.Module):
    def __init__(self, in_channels, class_num):
        super(CTCPredictor, self).__init__()
        self.class_num = class_num
        self.clf = nn.Linear(in_channels, self.class_num)

    def forward(self, x, targets=None):
        x = self.clf(x)
        if self.training:
            x = F.log_softmax(x, dim=-1).permute(1,0,2)
            # print(targets.shape)
            input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
            if targets.shape[0]==0:
                return x.sum()*0
            target_lengths, targets_sum = self.prepare_targets(targets)
            
            # print(x.shape,targets.shape,target_lengths.shape)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True) / 10
            loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)/2
            return loss
        return x
    def prepare_targets(self, targets):
        target_lengths = (targets != self.class_num - 1).long().sum(dim=-1) # has bugs, maybe not iterrable, next line broken
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        sum_targets = torch.cat(sum_targets)
        return target_lengths, sum_targets
class ImageS2SM(nn.Module):
    def __init__(self, in_channels, out_channels,bidirectional=True,use_len_embed=False,use_res_link=False,use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(ImageS2SM, self).__init__()
        # conv_func = conv_with_kaiming_uniform(True, True, False, False)
        # convs = []
        self.use_rnn = use_rnn
        # for i in range(2):
        #     convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        # self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, 256, out_channels) if self.use_rnn else nn.Linear(in_channels,out_channels)
        self.use_len_embed = use_len_embed
        if self.use_len_embed:
            self.len_embed = nn.Embedding(20, in_channels)
    def forward(self, x, lens=None, dictionary=None):
        #N,W,C
        x = x.permute(1, 0, 2)  # WxNxC
        if self.use_len_embed:
            # print("hello")
            x2 = self.len_embed(lens)
            x = x + x2[None]
        # print("before i-rnn:", x.shape)
        x = self.rnn(x)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        return x

class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
class FontEmbedding(nn.Module):
    def __init__(self,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      max_length=10,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True,
                      use_res_link=False,
                      use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(FontEmbedding, self).__init__()
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        self.use_rnn = use_rnn
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_encoder = nn.Sequential(
            conv_func(3, 64, 3, stride=(1, 1)),
            conv_func(64, 64, 3, stride=(2, 2)),
            nn.MaxPool2d(2, 2),
            conv_func(64, 128, 3, stride=(1, 1)),
            conv_func(128, 128, 3, stride=(2, 2)),
            nn.MaxPool2d(2, 2),
            conv_func(128, 256, 3, stride=(1, 1)),
            conv_func(256, 256, 3, stride=(2, 2)),
            nn.MaxPool2d(2, 2),
        )
        # self.rnn = nn.LSTM(char_vector_dim, out_channels,num_layers=1,bidirectional=bidirectional)
        # self.rnn = BidirectionalLSTM(char_vector_dim, 256, out_channels) if self.use_rnn else nn.Linear(char_vector_dim,out_channels)
        self.fonts = load_fonts(self.lexicon,"fonts/font-images",size=(64,64))
        self.font_num = self.fonts.size(0)
        # print(self.fonts.shape)
    def forward(self,inputs):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        # print(inputs)
        embeddings_batch = []
        font_embeddings = self.char_encoder(self.fonts.to(inputs[0].device)).view(self.font_num, -1)
        return font_embeddings
class WordEmbeddingModule(nn.Module):
    def __init__(self,
                      cfg,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      max_length=10,
                      embedding_group=1,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True,
                      use_res_link=False,
                      use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(WordEmbeddingModule, self).__init__()
        self.use_rnn = use_rnn
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.embedding_group = embedding_group
        self.char_embedding = nn.Embedding(len(self.lexicon), embedding_dim*self.embedding_group)
        self.use_font_embedding = False
        self.use_padding = cfg.MODEL.ALIGN.WORDEMBEDDING.USE_PADDING
        if self.use_padding:
            self.padding_embedding = nn.Embedding(1, embedding_dim)
        if self.use_font_embedding:
            self.font_embedding = FontEmbedding(lexicon = lexicon)
    def forward(self,inputs):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        # print(inputs)
        embeddings_batch = []
        if self.use_font_embedding:
            font_embeddings = self.font_embedding(inputs)
        for idx,word in enumerate(inputs):
            # print(idx, word)
            assert len(word)>0,word
            if self.use_font_embedding:
                embeddings = (self.char_embedding(word) + font_embeddings[word])/2
            else:
                embeddings = self.char_embedding(word)
                embeddings = embeddings.reshape([-1, self.embedding_dim]) # [char_num*eg, ed]
                # print(len(word), embeddings.shape)
            if self.use_padding and embeddings.size(0) < self.max_length:
                paddings = self.padding_embedding(torch.tensor([0]*(self.max_length - embeddings.size(0))).type_as(word))
                # print(embeddings.shape, paddings.shape)
                feat = torch.cat([embeddings, paddings])[None, None]
                # print(feat.shape)
            else:
                feat = nn.functional.interpolate(
                    embeddings[None,None,...], 
                    size=(self.max_length,self.embedding_dim), 
                    mode='bilinear', 
                    align_corners=True)
            embeddings_batch.append(feat)
        embeddings_batch = torch.cat(embeddings_batch,dim=1)[0] # [b, self.max_length, embedding_dim]
        return embeddings_batch
class WordEmbeddingTextLenSensitiveModule(nn.Module):
    def __init__(self,
                      cfg,
                      output_size_list, lens_area,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True,
                      use_res_link=False,
                      use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(WordEmbeddingTextLenSensitiveModule, self).__init__()
        self.use_rnn = use_rnn
        self.output_size_list = output_size_list
        self.lens_area = lens_area
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_embedding = nn.Embedding(len(self.lexicon), embedding_dim)
        self.use_font_embedding = False
        self.use_padding = cfg.MODEL.ALIGN.WORDEMBEDDING.USE_PADDING
        if self.use_padding:
            self.padding_embedding = nn.Embedding(1, embedding_dim)
        if self.use_font_embedding:
            self.font_embedding = FontEmbedding(lexicon = lexicon)
    def map_to_lens(self,len_):
        for idx, (len_max,size_) in enumerate(zip(self.lens_area, self.output_size_list)):
            if len_ <= len_max:
                return idx,size_
    def forward(self,inputs,texts):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        # print(inputs)
        embeddings_batch = [[] for i in range(len(self.lens_area))]
        texts_batch = [[] for i in range(len(self.lens_area))]
        for idx,word in enumerate(inputs):
            # print(idx, word)
            assert len(word)>0,word
            emb_idx, size_ = self.map_to_lens(len(word))
            embeddings = self.char_embedding(word)

            feat = nn.functional.interpolate(
                embeddings[None,None,...], 
                size=(size_,self.embedding_dim), 
                mode='bilinear', 
                align_corners=True)
            embeddings_batch[emb_idx].append(feat)
            texts_batch[emb_idx].append(texts[idx])
        embeddings_batch = [torch.cat(x,dim=1)[0] for x in embeddings_batch if len(x)>0] # [b, self.max_length, embedding_dim]
        return embeddings_batch, [v for v in texts_batch if len(v) > 0]
class TextS2SM(nn.Module):
    def __init__(self,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      max_length=10,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True,
                      use_len_embed=False,
                      use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(TextS2SM, self).__init__()
        self.use_rnn = use_rnn
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_encoder = nn.Sequential(
            nn.Linear(embedding_dim, char_vector_dim),
            nn.ReLU(inplace=True)
        )
        # self.rnn = nn.LSTM(char_vector_dim, out_channels,num_layers=1,bidirectional=bidirectional)
        self.rnn = BidirectionalLSTM(char_vector_dim, 256, out_channels) if self.use_rnn else nn.Linear(char_vector_dim,out_channels)
        self.use_len_embed = use_len_embed
        if self.use_len_embed:
            self.len_embed = nn.Embedding(20, char_vector_dim)
        
    def forward(self,embeddings_batch,lens=None):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        char_vector = self.char_encoder(embeddings_batch)
        char_vector = char_vector.permute(1, 0, 2).contiguous() # [w, b, c]
        if self.use_len_embed:
            x2 = self.len_embed(lens)
            char_vector = char_vector + x2[None]

        x = self.rnn(char_vector)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels=256):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.decoder = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(self.in_channels, self.in_channels)
        )
    def l2_loss(self, embedding1, embedding2):
        e1 = embedding1.view(embedding1.size(0),-1)
        e2 = embedding2.view(embedding2.size(0),-1)
        loss = (((e1-e2)**2).sum(dim=1)**0.5).mean()
        return loss
    def forward(self,img_f, word_f):
        img_f = self.decoder(img_f)
        # print(img_f.shape, word_f.shape)
        loss = self.l2_loss(img_f, word_f)
        return loss*0.1
class CharCountHead(nn.Module):
    def __init__(self, in_channels=128, size=(4, 15), class_num=20):
        super(CharCountHead, self).__init__()
        self.class_num = class_num
        self.in_channels = in_channels
        self.h, self.w = size
        conv_func = conv_with_kaiming_uniform(True, True, use_deformable=False, use_bn=False)
        self.conv = nn.Sequential(
            conv_func(in_channels, in_channels, 3, stride=(2, 1)),
            conv_func(in_channels, in_channels//2, 3, stride=(2, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.in_channels//2*self.w, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.in_channels, class_num)
        )
        self.loss = nn.CrossEntropyLoss()
    def wrap_to_max(self, x):
        y = torch.zeros(x.size(0), self.in_channels//2*self.w).type_as(x)
        # print(y.shape, x.shape)
        y[:,:x.size(1)] = x
        return y
    def forward(self, embeddings, targets=None):
        feat = self.conv(embeddings)
        new_feat = self.wrap_to_max(feat.view(feat.size(0), -1))
        nums = self.classifier(new_feat)
        if self.training:
            loss = self.loss(nums, targets)
            return loss
        else:
            return nums

        # print(embeddings.shape, feat.shape, nums.shape, loss)
class AlignHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AlignHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        resolution = cfg.MODEL.ALIGN.POOLER_RESOLUTION
        canonical_scale = cfg.MODEL.ALIGN.POOLER_CANONICAL_SCALE
        # print(resolution)
        self.use_only_spotter = cfg.MODEL.ALIGN.USE_ONLY_SPOTTER
        self.use_ctc_loss = cfg.MODEL.ALIGN.USE_CTC_LOSS
        self.use_no_rnn = cfg.MODEL.ALIGN.USE_NO_RNN
        self.use_domain_align_loss = cfg.MODEL.ALIGN.USE_DOMAIN_ALIGN_LOSS
        self.is_chinese = cfg.MODEL.ALIGN.IS_CHINESE
        self.use_word_aug = cfg.MODEL.ALIGN.USE_WORD_AUG
        self.use_partial_samples = cfg.MODEL.ALIGN.USE_PARTIAL_SAMPLES
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.use_poly = cfg.MODEL.FCOS.USE_POLY

        self.use_char_count = cfg.MODEL.ALIGN.USE_CHAR_COUNT
        self.use_char_count_detach = cfg.MODEL.ALIGN.USE_CHAR_COUNT_DETACH
        self.use_text_feat = cfg.MODEL.ALIGN.WORDEMBEDDING.USE_TEXT_FEAT
        self.use_cut_word = cfg.MODEL.ALIGN.WORDEMBEDDING.USE_CUT_WORD
        self.use_contrastive_loss = cfg.MODEL.ALIGN.USE_CONTRASTIVE_LOSS

        self.use_segment_embedding = cfg.MODEL.ALIGN.WORDEMBEDDING.USE_SEGMENT_EMBEDDING
        if self.use_poly:
            if self.use_segment_embedding:
                self.pooler = PolyPoolerTextLenSensitive(
                    num_points=7,
                    output_size_list=[(4,8),(4,12),(4,16),(4,20)],
                    lens_area = [4,8,12,1000],
                    scales=self.scales,
                    sampling_ratio=1,
                    canonical_scale=canonical_scale,
                    mode='align')
            else:
                self.pooler = PolyPooler(
                    num_points=7,
                    output_size=resolution,
                    scales=self.scales,
                    sampling_ratio=1,
                    canonical_scale=canonical_scale,
                    mode='align')
        else:
            self.pooler = Pooler(
                output_size=resolution,
                scales=self.scales,
                sampling_ratio=1,
                canonical_scale=canonical_scale,
                mode='alignadaptive')
        
        out_channels = 128
        in_channels = 256
        if self.is_chinese:
            # lexicon = np.load("./datasets/rctw/chars.npy").tolist()
            lexicon = load_chars()
            # print(len(lexicon))
            self.text_generator = TextGenerator(ratios=[1,0,1,5],chars=lexicon)
        else:
            #ratios=[1,1,1,5]
            #ratios=[1,1,1,9] 77.88
            self.text_generator = TextGenerator()
            # self.text_generator = TextGenerator(sim_fn='cossim')
        # self.box_augumentor = make_box_aug()
        self.use_len_embed = cfg.MODEL.ALIGN.USE_LEN_EMBED
        self.image_s2sm = ImageS2SM(in_channels, out_channels,bidirectional=True,use_len_embed=self.use_len_embed, use_rnn = (False if self.use_no_rnn else True))
        if not self.use_only_spotter:
            if self.use_text_feat:
                self.word_embedding = TextFeat()
            else:
                if self.use_segment_embedding:
                    self.word_embedding = WordEmbeddingTextLenSensitiveModule(cfg,
                                output_size_list=[8,12,16,20], lens_area= [4,8,12,1000],
                                out_channels=out_channels,
                                embedding_dim=256,
                                char_vector_dim=256,
                                lexicon = self.text_generator.chars,
                                bidirectional=True,
                                use_rnn = (False if self.use_no_rnn else True))
                else:
                    self.word_embedding = WordEmbeddingModule(cfg, out_channels=out_channels,
                                embedding_group=cfg.MODEL.ALIGN.WORDEMBEDDING.EMBEDDING_GROUP,
                                embedding_dim=256,
                                char_vector_dim=256,
                                max_length=resolution[1],
                                lexicon = self.text_generator.chars,
                                bidirectional=True,
                                use_rnn = (False if self.use_no_rnn else True))
            self.text_s2sm = TextS2SM(out_channels=out_channels,
                        embedding_dim=256,
                        char_vector_dim=256,
                        max_length=resolution[1],
                        lexicon = self.text_generator.chars,
                        bidirectional=True,
                        use_len_embed=self.use_len_embed,
                        use_rnn = (False if self.use_no_rnn else True))
            self.decoder = Decoder()
        conv_func_defor = conv_with_kaiming_uniform(True, True, use_deformable=True, use_bn=False)
        conv_func = conv_with_kaiming_uniform(True, True, use_deformable=False, use_bn=False)
        convs = []
        # self.use_rnn = use_rnn
        #添加defor无法work
        # convs.append(conv_func_defor(in_channels, in_channels, 3))
        # convs.append(conv_func_defor(in_channels, in_channels, 3))
        self.use_scalenet = cfg.MODEL.ALIGN.USE_SCALENET
        if self.use_scalenet:
            if self.use_segment_embedding:
                self.rois_conv = AttentionRecognitionHead(4,in_channels,256,20)
            else:
                self.rois_conv = ScaleNet(in_channels,size=resolution)
        else:
            for i in range(2):
                # convs.append(conv_func_defor(in_channels, in_channels, 3))
                convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
            self.rois_conv = nn.Sequential(*convs)
        if self.use_char_count:
            size = (4,20) if self.use_segment_embedding else resolution
            self.char_count = CharCountHead(in_channels, size=size)
        frames = resolution[1]
        if self.use_ctc_loss:
            self.ctc_head = CTCPredictor(out_channels,len(self.text_generator.chars)+1)
        self.feat_dim = 128*frames 
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
    # @torch.no_grad()
    # def get_word_embedding(self,texts,device):
    #     words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in texts]
    #     words_embedding = self.word_embedding(words).detach()
    #     return words_embedding
    def show_rois(self, images, polys, nums, path):
        image_name = os.path.basename(str(path))
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0].copy()
        # print(image_de)
        for idx, (poly, num) in enumerate(zip(polys, nums)):
            image_temp = image_de.copy()
            poly = poly.reshape([-1,14,2]).astype(np.int32)
            for p in poly:
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                cv2.drawContours(image_temp, [p], -1, thickness=2, color=color)
            cv2.imwrite('./temp/{}_{}_{}.jpg'.format(image_name, idx, num), image_temp)
        # for idx, (poly, num) in enumerate(zip(polys, nums)):
        #     minx, miny, maxx, maxy = int(np.min(poly[::2])), int(np.min(poly[1::2])), int(np.max(poly[::2])), int(np.max(poly[1::2]))
        #     roi = image_de[miny:maxy, minx:maxx]
        #     cv2.imwrite('./temp/{}_{}_{}.jpg'.format(image_name, idx, num), roi)
            # cv2.imwrite('./temp/{}'.format(image_name), image_de)


    @torch.no_grad()
    def get_image_embedding(self,proposals_per_im,x,img_lens=None, is_partial=False, images=None):
        select_boxes = []
        if 'polys' in proposals_per_im.fields():
            polys = proposals_per_im.get_field('polys')
        # print(polys.shape)
        if proposals_per_im.bbox.size(0) == 0:
            imgs_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
            ver = torch.zeros([0,0,2]).type_as(x[0])
            att = torch.zeros([0,15,15]).type_as(x[0])
        else:
            rois = self.pooler(x, [proposals_per_im])
            # if self.use_char_count and is_partial:
            if is_partial:
                polys_np = polys.data.cpu().numpy()
                # preds = self.char_count_head(rois)
                lens = proposals_per_im.get_field('lens')
                nums = lens.data.cpu().numpy()
                scores = proposals_per_im.get_field('scores').view(-1)
                all_partial_polys = []
                all_query_lens = []
                # self.show_rois(images, polys_np, nums, proposals_per_im.get_field('path'))
                for idx,num in enumerate(nums):
                    if num <= 3:
                        continue
                    partial_polys, query_lens = self.get_all_partial_proposals(polys_np[idx], text_len=num)
                    all_partial_polys.append(partial_polys)
                    all_query_lens.append(query_lens)
                # self.show_rois(images, [p.data.cpu().numpy() for p in all_partial_polys], nums, proposals_per_im.get_field('path'))
                if len(all_partial_polys)!= 0:
                    # print("part",all_partial_polys)
                    # print("full",polys)
                    all_partial_polys = torch.cat(all_partial_polys).type_as(polys)
                    all_query_lens = torch.cat([lens, torch.cat(all_query_lens).type_as(lens)])
                    # print("polys", polys)
                    # print("partial_polys", all_partial_polys)
                    # print(all_partial_polys.shape, torch.cat([polys, all_partial_polys]).shape)
                    all_polys = torch.cat([polys, all_partial_polys])
                    minx,miny,maxx,maxy = torch.min(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                          torch.min(all_polys[:,1::2], dim=1, keepdim=True)[0],\
                                          torch.max(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                          torch.max(all_polys[:,1::2], dim=1, keepdim=True)[0]
                    bbox = torch.cat([minx,miny,maxx,maxy],dim=1)
                    proposals_per_im.bbox = bbox
                    proposals_per_im.add_field('polys', torch.cat([polys, all_partial_polys]))
                    proposals_per_im.add_field('query_lens', all_query_lens)
                    rois = self.pooler(x, [proposals_per_im])
                    # import ipdb; ipdb.set_trace()
                    # proposals_per_im.add_field('rois', rois)
            # else:
            #     rois = self.pooler(x, [proposals_per_im])
                
            if self.use_scalenet:
                re, att = self.rois_conv(rois)
            else:
                re = self.rois_conv(rois)
                att = torch.zeros([0,15,15]).type_as(x[0])
            re = re.mean(dim=2).permute(0,2,1).contiguous()
            
            imgs_embedding = self.image_s2sm(re, lens=img_lens)
            # select_boxes.append(proposals_per_im.bbox)
            k = 1
            imgs_embedding_nor = nn.functional.normalize((imgs_embedding*k).tanh().view(imgs_embedding.size(0),-1))
        return imgs_embedding_nor, att, select_boxes
    def compute_domain_align_loss(self,embedding1, embedding2):
        embedding1 = embedding1.tanh().view(embedding1.size(0),-1)
        embedding2 = embedding2.tanh().view(embedding2.size(0),-1)
        loss = (((embedding1-embedding2)**2).sum(dim=1)**0.5).mean()
        return loss
    def l2_loss(self, embedding1, embedding2):
        e1 = embedding1.view(embedding1.size(0),-1)
        e2 = embedding2.view(embedding2.size(0),-1)
        loss = (((e1-e2)**2).sum(dim=1)**0.5).mean()
        return loss

    def compute_loss(self, embedding1, embedding2, targets=None, words1=None,words2=None, using_mask=False):
        k = 1
        iou = self.compute_similarity(embedding1, embedding2, k)
        # similarity =  self.text_generator.calculate_similarity_matric(words1, words2) if targets==None else targets
        similarity = targets.type_as(iou) #torch.tensor(similarity).type_as(iou) if targets==None else similarity
        # if using_mask:
        #     similarity = (similarity - self.sim_thred)/(1-self.sim_thred)
        if not self.use_contrastive_loss:
            loss = self.sim_loss_func(iou, similarity)
            loss = loss.max(dim=1)[0].mean()
        else:
            loss = matric_similarity_learning_loss(iou, similarity)
        return loss
    def compute_similarity(self,embedding1, embedding2,k=1):
        embedding1_nor = nn.functional.normalize((embedding1*k).tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize((embedding2*k).tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        return inter
    
    def get_augmented_words(self, texts, device):
        word_texts = texts.copy()
        if self.use_word_aug:
            word_texts.extend([self.text_generator(text) for text in texts])
        if not self.is_chinese:
            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in word_texts]
        else:
            words = [torch.tensor(self.text_generator.label_map(text)).long().to(device) for text in word_texts]
        return word_texts, words
    def char_count_head(self, features, words=None):
        if self.training:
            labels = torch.tensor([min(len(v), 19) for v in words]).to(features.device)
            return self.char_count(features, labels)
        else:
            return self.char_count(features)
    def generate_partial_proposals(self, poly, query_len, text_len=10, num_point=7):
        poly = poly.reshape([-1,2])
        point_num = len(poly)//2
        inter_up_x = (poly[point_num-1, 0] - poly[0, 0]) / (text_len*2)
        inter_up_y = (poly[point_num-1, 1] - poly[0, 1]) / (text_len*2)
        inter_do_x = (poly[point_num, 0] - poly[-1, 0]) / (text_len*2)
        inter_do_y = (poly[point_num, 1] - poly[-1, 1]) / (text_len*2)

        span_up_x = np.linspace(poly[0, 0], poly[0, 0]+inter_up_x*(query_len*2), num_point)
        span_up_y = np.linspace(poly[0, 1], poly[0, 1]+inter_up_y*(query_len*2), num_point)
        span_up = np.stack((span_up_x, span_up_y),axis=1)
        span_do_x = np.linspace(poly[-1, 0] + inter_do_x*(query_len*2), poly[-1, 0], num_point)
        span_do_y = np.linspace(poly[-1, 1] + inter_do_y*(query_len*2), poly[-1, 1], num_point)
        span_do = np.stack((span_do_x, span_do_y),axis=1)

        span = np.concatenate((span_up, span_do), axis=0)
        ret = [span]
        for i in range(1,(text_len-query_len)*2+1):
            if i%2 != 0:
                continue
            new_span = np.zeros_like(span)
            new_span[:num_point,0] = span[:num_point,0] + inter_up_x*i
            new_span[:num_point,1] = span[:num_point,1] + inter_up_y*i
            new_span[num_point:,0] = span[num_point:,0] + inter_do_x*i
            new_span[num_point:,1] = span[num_point:,1] + inter_do_y*i

            ret.append(new_span)
        return torch.tensor(np.array(ret).reshape([-1, num_point*4]))
    def get_all_partial_proposals(self, poly, text_len=10, num_point=7):
        assert text_len>3
        all_partial_polys = []
        query_lens = []
        # poly_np = poly.data.cpu().numpy()
        
        for query_len in range(2,text_len-1):
        # for query_len in range(2,min(text_len-1, 4)):
            partial_polys = self.generate_partial_proposals(poly, query_len, text_len, num_point)
            all_partial_polys.append(partial_polys)
            query_lens.extend([query_len]*partial_polys.size(0))
        all_partial_polys = torch.cat(all_partial_polys)
        query_lens = torch.tensor(query_lens).view(-1)
        return all_partial_polys, query_lens
    def training_loss(self,rois, imgs_texts, word_texts, imgs_embedding, words_embedding, re, we, phase=''):
        def wrap_name(name):
            return name + '{}'.format(phase)
        def sum_to_zero_loss(feats):
            loss = 0.0
            for f in feats:
                loss += f.sum() * 0
            return loss
        if rois.size(0)<2:
            # zero_loss = rois.sum()*0 + self.char_count_head(rois, imgs_texts).sum()*0 + words_embedding.sum()*0 + 
            zero_loss = sum_to_zero_loss([rois, imgs_embedding, words_embedding, re, we])
            # zero_loss = sum_to_zero_loss([rois, self.char_count_head(rois, imgs_texts), imgs_embedding, words_embedding, re, we])
            # loss = {"loss_wi":zero_loss,"loss_ww":zero_loss,"loss_ii":zero_loss,
                    # "loss_wivr":zero_loss,"loss_wwvr":zero_loss,"loss_iivr":zero_loss}
            loss = {wrap_name("loss_wi"):zero_loss,wrap_name("loss_ww"):zero_loss,wrap_name("loss_ii"):zero_loss}
            if self.use_char_count:
                loss[wrap_name('loss_cnt')] = zero_loss
            if self.use_domain_align_loss:
                loss[wrap_name("loss_da")] = zero_loss
                loss[wrap_name("loss_dab")] = zero_loss
            return loss

        # print(rois.shape,len(texts))
        assert rois.size(0) == len(imgs_texts),print(rois.size(0),len(imgs_texts))
        # similarity targets
        wi_sim = torch.tensor(self.text_generator.calculate_similarity_matric(word_texts, imgs_texts)).to(rois.device)
        ww_sim = torch.tensor(self.text_generator.calculate_similarity_matric(word_texts, word_texts)).to(rois.device)
        ii_sim = torch.tensor(self.text_generator.calculate_similarity_matric(imgs_texts, imgs_texts)).to(rois.device)

        loss_fn = self.compute_loss
        wi_loss = loss_fn(words_embedding.detach(), imgs_embedding, wi_sim, using_mask=True)
        ww_loss = loss_fn(words_embedding, words_embedding, ww_sim, using_mask=True)
        ii_loss = loss_fn(imgs_embedding, imgs_embedding, ii_sim, using_mask=True)
        
        # loss = {"loss_wi":wi_loss*10,"loss_ww":ww_loss*10,"loss_ii":ii_loss*10,
        #         "loss_wivr":wivr_loss*10,"loss_wwvr":wwvr_loss*10,"loss_iivr":iivr_loss*10}
        loss = {wrap_name("loss_wi"):wi_loss*10,wrap_name("loss_ww"):ww_loss*10,wrap_name("loss_ii"):ii_loss*10}
        # loss["loss_lc"] = self.compute_domain_align_loss(img_lf, word_lf[:len(texts),...].detach())
        # char count head
        if self.use_char_count:
            if self.use_char_count_detach:
                loss[wrap_name("loss_cnt")] = self.char_count_head(rois.detach(), imgs_texts)
            else:
                loss[wrap_name("loss_cnt")] = self.char_count_head(rois, imgs_texts)
        if self.use_ctc_loss:
            
            max_len = imgs_embedding.size(1)
            selected_idx = [i for i,text in enumerate(texts) if len(text)<max_len]
            selected_texts = [texts[i] for i in selected_idx]
            class_num = len(self.text_generator.chars)+1
            words = torch.tensor([self.text_generator.label_map_with_padding(text, max_len=max_len, padding=class_num-1) for text in selected_texts]).long().to(rois.device)
            loss[wrap_name("loss_rc")] = self.ctc_head(imgs_embedding[selected_idx], words)
        if self.use_domain_align_loss:
            if not self.use_mil:
                loss[wrap_name("loss_da")] = self.compute_domain_align_loss(imgs_embedding, words_embedding[:len(imgs_texts),...].detach())
                loss[wrap_name("loss_dab")] = self.decoder(re, we[:len(imgs_texts),...].detach())/4
            else:
                loss[wrap_name("loss_da")] = self.compute_domain_align_loss(imgs_embedding, words_embedding[self.full_idxs].detach())
                loss[wrap_name("loss_dab")] = self.decoder(re, we[self.full_idxs].detach())/4
        return loss
    def segment_train(self,x, samples):
        def update_loss_dict(target, inputs, weight=1):
            for k,v in inputs.items():
                if k not in target.keys():
                    target.update({k:v*weight})
                else:
                    target[k] += v*weight
            return target
        proposals = samples["retrieval_samples"]
        texts = []
        new_proposals = []
        losses = {}
        for proposals_per_im in proposals:
            texts.extend(proposals_per_im.get_field("texts").tolist())
            new_proposals.append(proposals_per_im)
        
        rois_batch, img_texts_batch = self.pooler(x, new_proposals, texts)
        
        

        word_texts, words = self.get_augmented_words(texts, rois_batch[0].device)

        wes, word_texts_batch = self.word_embedding(words, word_texts)
        # for v in rois_batch:
            # print(v.shape)
        # res = [self.rois_conv(v).mean(dim=2).permute(0,2,1).contiguous() for v in rois_batch]
        if self.use_scalenet:
            res = [self.rois_conv(v, (v.size(-1)-8)/4) for v in rois_batch]
        else:
            res = [self.rois_conv(v).mean(dim=2).permute(0,2,1).contiguous() for v in rois_batch]
        total_num = 0
        for rois, we, re, img_text_per_seg, word_text_per_seg in zip(rois_batch, wes, res, img_texts_batch, word_texts_batch):
            sample_num = rois.size(0)
            total_num += sample_num
            imgs_embedding = self.image_s2sm(re)
            words_embedding = self.text_s2sm(we)
            # print(imgs_embedding.shape, words_embedding.shape, rois.shape, len(img_text_per_seg), len(word_text_per_seg))
            assert len(img_text_per_seg)==len(word_text_per_seg), (img_text_per_seg, word_text_per_seg, word_texts,img_texts_batch,word_texts_batch)
            loss = self.training_loss(rois, img_text_per_seg, word_text_per_seg, imgs_embedding, words_embedding, re, we)
            # print("loss: ",loss)
            losses = update_loss_dict(losses, loss, weight=sample_num)
            # losses.update(loss)
        # print("losses: ",losses)
        if total_num != 0:
            losses = {k:v/total_num for k,v in losses.items()}
        return None,losses
    def segment_test(self,x, samples,is_words):
        proposals = samples["retrieval_samples"]
        for proposals_per_im in proposals:
            # print(proposals_per_im.fields())
            if not self.is_chinese:
                idxs, texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
            else:
                texts = proposals_per_im.get_field("texts").tolist()
                # print("here:", texts)
            if is_words:
                
                if len(texts) == 0:
                    words_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
                else:
                    if not self.is_chinese:
                        words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x[0].device) for text in texts]
                    else:
                        words = [torch.tensor(self.text_generator.label_map(text)).long().to(x[0].device) for text in texts]

                    # wes, word_texts_batch = self.word_embedding(words, texts)
                    wes = [self.text_s2sm(self.word_embedding([word], [text])[0][0]) for word, text in zip(words, texts)]
                    # words_embedding = self.text_s2sm(we)
                    words_embedding_nor = [nn.functional.normalize(v.tanh().view(v.size(0),-1)) for v in wes]
                    proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)
            # print(proposals_per_im.get_field('lens'))
            rois_batch, img_texts_batch = self.pooler(x, [proposals_per_im], [''.join(['a']*v) for v in proposals_per_im.get_field('lens')])
            if self.use_scalenet:
                res = [self.rois_conv(v, (v.size(-1)-8)/4) for v in rois_batch]
            else:
                res = [self.rois_conv(v).mean(dim=2).permute(0,2,1).contiguous() for v in rois_batch]
            imgs_embedding_nor = [nn.functional.normalize(self.image_s2sm(v).tanh().view(v.size(0),-1)) for v in res]
            # imgs_embedding_nor, att, select_boxes = self.get_image_embedding(proposals_per_im,x, is_partial=False)
            # imgs_embedding_nor, att, select_boxes = self.get_image_embedding(proposals_per_im,x, is_partial=True, images=images)
            proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
            # if self.use_scalenet:
            #     proposals_per_im.add_field("attention", att)
            # if is_words:
            #     proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)
            #     # proposals_per_im.add_field("words_embedding_nor", words_embedding)
            #     # proposals_per_im.add_field("word_lf", word_lf)

        return proposals, {"select_boxes":None}
    def train_spotter(self, x, samples):
        if self.training:
            proposals = samples["retrieval_samples"]
            texts = []
            new_proposals = []
            loss = {}
            for proposals_per_im in proposals:
                texts.extend(proposals_per_im.get_field("texts").tolist())
                new_proposals.append(proposals_per_im)
            
            rois = self.pooler(x, new_proposals)
            re = self.rois_conv(rois).mean(dim=2).permute(0,2,1).contiguous()
            

            imgs_embedding = self.image_s2sm(re)
            max_len = imgs_embedding.size(1)
            selected_idx = [i for i,text in enumerate(texts) if len(text)<max_len]
            selected_texts = [texts[i] for i in selected_idx]
            class_num = len(self.text_generator.chars)+1
            words = torch.tensor([self.text_generator.label_map_with_padding(text, max_len=max_len, padding=class_num-1) for text in selected_texts]).long().to(rois.device)
            loss["loss_rc"] = self.ctc_head(imgs_embedding[selected_idx], words)
            return None,loss
    def test_spotter(self, x, samples):
        proposals = samples["retrieval_samples"]
        if proposals[0].bbox.size(0)==0:
            proposals[0].add_field("recognized_texts", [])
            return proposals, None
        rois = self.pooler(x, proposals)
        re, att = self.rois_conv(rois)
        re = re.mean(dim=2).permute(0,2,1).contiguous()
        imgs_embedding = self.image_s2sm(re)
        preds = self.ctc_head(imgs_embedding)
        _, preds = preds.max(2)
        texts = []
        for pred in preds:
            pred = pred.contiguous().view(-1)
            pred_size = torch.IntTensor([pred.size(0)])
            text = self.text_generator.decode(pred.data, pred_size.data, raw=False)
            texts.append(text)
        proposals[0].add_field("recognized_texts", texts)
        return proposals,None
    def mil_loss(self, imgs_embedding, words_embedding, proposals):
        offset = 0
        for proposals_per_im in proposals:
            bag_ids = proposals_per_im.get_field("bag_ids")
            full_num, all_num = bag_ids[0], bag_ids[-1]
            if full_num == all_num: continue # no partial samples
            partial_query_word = words_embedding[offset+full_num: offset+all_num]
            partial_image_word = imgs_embedding[offset+full_num: offset+all_num]
            sim = self.compute_similarity(partial_query_word, partial_image_word)
            bag_sim = sim.max(dim=1)[0]
            print(sim.shape)
            print(bag_sim)



            offset += all_num
    def split_full_partial(self, proposals, device):
        max_nums = sum([p.get_field("bag_ids")[-1] for p in proposals])
        idxs = torch.tensor([-1]*max_nums).to(device)
        offset = 0
        for idx, proposals_per_im in enumerate(proposals):
            bag_ids = proposals_per_im.get_field("bag_ids")
            full_num, all_num = bag_ids[0], bag_ids[-1]
            idxs[offset:offset+full_num] = idx*2
            idxs[offset+full_num:offset+all_num] = idx*2+1
            offset += all_num
        return idxs
    def forward(self, x, samples,images=None,is_words=None):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        if self.training:
            if self.use_only_spotter:
                return self.train_spotter(x, samples)
            if self.use_segment_embedding:
                return self.segment_train(x, samples)
            proposals = samples["retrieval_samples"]
            texts = []
            new_proposals = []
            for proposals_per_im in proposals:
                texts.extend(proposals_per_im.get_field("texts").tolist())
                if self.use_partial_samples:
                    texts.extend(proposals_per_im.get_field("partial_texts"))
                    all_polys = torch.cat([proposals_per_im.get_field('polys'), proposals_per_im.get_field('partial_polys')])
                    minx,miny,maxx,maxy = torch.min(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                          torch.min(all_polys[:,1::2], dim=1, keepdim=True)[0],\
                                          torch.max(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                          torch.max(all_polys[:,1::2], dim=1, keepdim=True)[0]
                    bbox = torch.cat([minx,miny,maxx,maxy],dim=1)
                    proposals_per_im.bbox = bbox
                    proposals_per_im.add_field('polys', all_polys)
                new_proposals.append(proposals_per_im)
            
            rois = self.pooler(x, new_proposals)
            
            
            imgs_texts = texts.copy()
            if self.use_cut_word:
                # print("before", texts)
                texts = [w for w in self.text_generator.cut_words(texts) if len(w)>1]
                # print("after", texts)
            # print(len(imgs_texts), len(texts))
            word_texts, words = self.get_augmented_words(texts, rois.device)
            
            if self.use_text_feat:
                we = self.word_embedding(word_texts)
            else:
                we = self.word_embedding(words)
            re = self.rois_conv(rois).mean(dim=2).permute(0,2,1).contiguous()
            
            if self.use_len_embed:
                img_lens = torch.tensor([min(len(v), 19) for v in imgs_texts]).long().to(re.device)
                word_lens = torch.tensor([min(len(v), 19) for v in word_texts]).long().to(we.device)
            else:
                img_lens, word_lens = None, None
            imgs_embedding = self.image_s2sm(re, lens=img_lens)
            words_embedding = self.text_s2sm(we, lens=word_lens)
            
            self.use_mil = False
            if self.use_mil:
                idxs = self.split_full_partial(proposals, device=re.device)
                full_idxs = torch.nonzero(idxs%2==0).view(-1)
                self.full_idxs = full_idxs
                full_idxs_list = full_idxs.data.cpu().numpy()

                loss = self.training_loss(rois[full_idxs], \
                                          [imgs_texts[v] for v in range(len(imgs_texts)) if v in full_idxs_list],\
                                          word_texts,\
                                          imgs_embedding[full_idxs],\
                                          words_embedding, re[full_idxs], we)
                # self.mil_loss(imgs_embedding, words_embedding, proposals)
            else:
                loss = self.training_loss(rois, imgs_texts, word_texts, imgs_embedding, words_embedding, re, we)
            
            return None,loss
        else:
            if self.use_only_spotter:
                return self.test_spotter(x, samples)
            if self.use_segment_embedding:
                return self.segment_test(x, samples,is_words)
            # is_words = True
            # select_boxes = []
            proposals = samples["retrieval_samples"]
            # print("is chinese:", self.is_chinese)
            for proposals_per_im in proposals:
                if not self.is_chinese:
                    idxs, texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                else:
                    texts = proposals_per_im.get_field("texts").tolist()
                    # print("here:", texts)
                if is_words:
                    
                    if len(texts) == 0:
                        words_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
                    else:
                        if not self.is_chinese:
                            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x[0].device) for text in texts]
                        else:
                            words = [torch.tensor(self.text_generator.label_map(text)).long().to(x[0].device) for text in texts]
                            # print(len(words))
                        # words_embedding, word_lf = self.word_embedding(words)
                        # print(words)
                        if self.use_len_embed:
                            img_lens = proposals_per_im.get_field("lens")
                            word_lens = torch.tensor([min(len(v), 19) for v in texts]).long().to(x[0].device)
                        else:
                            img_lens, word_lens = None, None
                        if self.use_text_feat:
                            we = self.word_embedding(texts)
                        else:
                            we = self.word_embedding(words)
                        words_embedding = self.text_s2sm(we, lens=word_lens)
                        k = 1
                        words_embedding_nor = nn.functional.normalize((words_embedding*k).tanh().view(words_embedding.size(0),-1))
                        # words_embedding_nor = nn.functional.normalize(words_embedding.view(words_embedding.size(0),-1))
                imgs_embedding_nor, att, select_boxes = self.get_image_embedding(proposals_per_im,x,img_lens=img_lens, is_partial=False)
                # imgs_embedding_nor, att, select_boxes = self.get_image_embedding(proposals_per_im,x, is_partial=True, images=images)
                proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
                # if self.use_scalenet:
                #     proposals_per_im.add_field("attention", att)
                # proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding)
                # proposals_per_im.add_field("ver", ver)
                # proposals_per_im.add_field("imgs_lf", img_lf)
                if is_words:
                    proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)
                    # proposals_per_im.add_field("words_embedding_nor", words_embedding)
                    # proposals_per_im.add_field("word_lf", word_lf)

            return proposals, {"select_boxes":select_boxes}

def denormalize(image):
    std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
    mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
    image.mul_(std_).add_(mean_)
    return image
import os
import numpy as np
from PIL import Image, ImageDraw
def vis_pss_map(img, pss, ori_h, ori_w):
    im = img.copy()
    img = Image.fromarray(im).convert('RGB').resize((ori_w, ori_h))
    pss_img = Image.fromarray((pss*255).astype(np.uint8)).convert('RGB').resize((ori_w, ori_h))
    pss_img = Image.blend(pss_img, img, 0.5)
    return pss_img
def vis_multi_image(image_list, shape=[1,-1]):
    image_num = len(image_list)
    h, w,_ = np.array(image_list[0]).shape
    #print h,w
    num_w = int(image_num/shape[0])
    num_h = shape[0]
    new_im = Image.new('RGB', (num_w*w,num_h*h))
    for idx, image in enumerate(image_list):
        idx_w = idx%num_w
        idx_h = int(idx/num_w)
        new_im.paste(image, (int(idx_w*w),int(idx_h*h)))
    return new_im
class AlignModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels,proposal_matcher):
        super(AlignModule, self).__init__()

        self.cfg = cfg.clone()
        
        self.detector = build_fcos(cfg, in_channels)
        self.proposal_matcher = proposal_matcher
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.use_box_aug = cfg.MODEL.ALIGN.USE_BOX_AUG
        self.use_retrieval = cfg.MODEL.ALIGN.USE_RETRIEVAL
        self.det_score = cfg.MODEL.ALIGN.DET_SCORE
        if self.use_retrieval:
            self.head = AlignHead(cfg, in_channels)
        self.batch_size_per_image = 256 
        self.positive_fraction = 0.25
        self.use_textness = cfg.MODEL.ALIGN.USE_TEXTNESS
    def visual(self,images,boxes):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)
        maps = boxes.view(-1,1).float()
        maps = maps.data.cpu().numpy()
        # print(maps.shape)
        nums = [6400,8000,8400,8500,8525]
        maps = [maps[:6400,:].reshape([80,80]),maps[6400:8000,:].reshape([40,40]),maps[8000:8400,:].reshape([20,20]),maps[8400:8500,:].reshape([10,10]),maps[8500:,:].reshape([5,5])]
        img_list = [Image.fromarray(image_de[0]).convert('RGB')]
        for single_map in maps:
            # single_map.reshape()
            img = vis_pss_map(image_de[0], single_map, 640,640)
            img_list.append(img)
        new_img = vis_multi_image(img_list,shape=[2,-1])
        img_path = os.path.join('temp','img_{}.jpg'.format(np.random.randint(0,999)))
        print(img_path)
        new_img.save(img_path)

        return None
    def test_visual(self,images,boxes,image_name):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0]
        boxes = boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        image = image_de.copy()
        cv2.drawContours(image, boxes, -1, color=(255,0,0), thickness=1)
        # if len(select_boxes)>0:
        #     boxes = select_boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        #     image2 = image_de.copy()
        #     cv2.drawContours(image2, boxes, -1, color=(255,0,0), thickness=1)
        #     image = np.concatenate((image, image2),axis=1)

        #     image = np.ascontiguousarray(image)
        # print(image.shape)
        img_path = os.path.join('temp',image_name)
        cv2.imwrite(img_path, image)

        
        

        return None
    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        return matched_idxs
    def select_positive_boxes(self, boxes, targets):
        new_boxes = []
        for boxes_per_image, targets_per_image in zip(boxes, targets):
            if boxes_per_image.bbox.size(0) == 0:
                boxes_per_image = targets_per_image
            matched_idxs = self.match_targets_to_proposals(
                boxes_per_image, targets_per_image
            )
            positive = torch.nonzero(matched_idxs >= 0).squeeze(1).data.cpu().numpy()
            positive_matched_targets = targets_per_image[positive]
            positive_boxes = boxes_per_image[positive]
            positive_boxes.add_field("texts", positive_matched_targets.get_field("texts"))
            positive_boxes = targets_per_image if positive_boxes.bbox.size(0)==0 else positive_boxes
            new_boxes.append(positive_boxes)
        return new_boxes
    def prepare_training_samples_for_textness_retrieval(self, boxes, targets):
        retrieval_samples = []
        textness_samples = []
        for boxes_per_image, targets_per_image in zip(boxes, targets):
            if boxes_per_image.bbox.size(0) == 0:
                boxes_per_image = targets_per_image
            matched_idxs = self.match_targets_to_proposals(
                boxes_per_image, targets_per_image
            )
            #retrieval_samples
            positive = torch.nonzero(matched_idxs >= 0).squeeze(1).data.cpu().numpy()
            positive_matched_targets = targets_per_image[matched_idxs[positive].data.cpu().numpy()]
            positive_boxes = boxes_per_image[positive]
            assert positive_matched_targets.bbox.size(0) == positive_boxes.bbox.size(0)
            
            positive_boxes = positive_boxes.clone_without_fields()
            positive_boxes.add_field("texts", positive_matched_targets.get_field("texts"))
            
            targets_per_image2 = targets_per_image.clone_without_fields()
            targets_per_image2.add_field("texts", targets_per_image.get_field("texts"))
            # print(positive_boxes.fields(),targets_per_image.fields())
            positive_boxes = targets_per_image2 if positive_boxes.bbox.size(0)==0 else cat_boxlist_texts([positive_boxes, targets_per_image2])
            # positive_boxes = targets_per_image2
            # positive_boxes.add_field("positive_num",torch.tensor([positive_boxes.bbox.size(0)]*(positive_boxes.bbox.size(0)+1)))
            # positive_boxes.add_field("positive_num",torch.tensor([positive_boxes.bbox.size(0)-targets_per_image2.bbox.size(0)]*(positive_boxes.bbox.size(0)+1)))
            retrieval_samples.append(positive_boxes)

            #textness_samples
            positive_boxes = positive_boxes.clone_without_fields()
            negative = torch.nonzero(matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD).squeeze(1)
            # print(negative.numel())
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # print(positive.shape[0])
            num_pos = min(positive_boxes.bbox.size(0), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            neg_idx_per_image = negative[perm2].data.cpu().numpy()
            
            negative_boxes = boxes_per_image[neg_idx_per_image].clone_without_fields()
            positive_boxes.add_field("textness",torch.ones([positive_boxes.bbox.size(0)], dtype=torch.uint8))
            negative_boxes.add_field("textness",torch.zeros([negative_boxes.bbox.size(0)], dtype=torch.uint8))
            # print(positive_boxes, negative_boxes)
            samples = cat_boxlist([positive_boxes, negative_boxes])
            textness_samples.append(samples)
        return retrieval_samples,textness_samples


    def forward(self, images, features, targets=None, vis=False,is_words=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
            vis (bool): visualise offsets

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        # print(targets)
        targets = [target.to(features[0].device) for target in targets]
        if self.training:
            # self.visual(images, boxes)
            rec_features = features[:len(self.scales)]
            boxes, losses = self.detector(images, features[1:], targets)
            if not self.use_retrieval:
                return None, losses
            if not self.use_textness:
                # _, loss_dict = self.head(rec_features, targets,images)
                # new_boxes = self.select_positive_boxes(boxes, targets) if self.use_box_aug else targets
                # print(boxes, targets)
                
                # new_boxes = self.select_positive_boxes(boxes, targets)
                # new_boxes = [cat_boxlist_texts([b, t]) for b, t in zip(new_boxes, targets)]
                new_boxes = targets
                _, loss_dict = self.head(rec_features, {"retrieval_samples":new_boxes},images)

                # self.visual(images, boxes)
                
                
                for k, v in loss_dict.items():
                    losses.update({k: v})
                return None, losses
            else:
                retrieval_samples,textness_samples = self.prepare_training_samples_for_textness_retrieval(boxes, targets)
                _, loss_dict = self.head(rec_features, {"retrieval_samples":retrieval_samples,"textness_samples":textness_samples},images)

                # self.visual(images, boxes)
                
                
                for k, v in loss_dict.items():
                    losses.update({k: v})
                return None, losses

        else:
            boxes, losses = self.detector(images, features[1:], targets)
            if not self.use_retrieval:
                # image_names = [os.path.basename(str(image.get_field("path"))) for image in targets]
                # self.test_visual(images, boxes[0].bbox,image_names[0])
                new_boxes = []
                for box, target in zip(boxes, targets):
                    scores = box.get_field("scores")
                    # pos_idxs = torch.nonzero(scores>0.08).view(-1)
                    # pos_idxs = torch.nonzero(scores>0.05).view(-1)
                    pos_idxs = torch.nonzero(scores>0.05).view(-1)
                    # print("0.2")
                    # pos_idxs = torch.nonzero(scores>0.2).view(-1)#75.43
                    # pos_idxs = torch.nonzero(scores>0.23).view(-1)#75.43
                    box = box[pos_idxs]
                    box.add_field("texts", target.get_field("texts"))
                    box.add_field("scale", target.get_field("scale"))
                    box.add_field("path", target.get_field("path"))
                    # box.add_field("y_trues", target.get_field("y_trues"))
                    new_boxes.append(box)
                return new_boxes, losses
            # print(images)
            # new_boxes = self.select_positive_boxes(boxes, targets)
            
            rec_features = features[:len(self.scales)]

            new_boxes = []
            for box, target in zip(boxes, targets):
                scores = box.get_field("scores")
                # pos_idxs = torch.nonzero(scores>0.08).view(-1)
                # pos_idxs = torch.nonzero(scores>0.05).view(-1)
                pos_idxs = torch.nonzero(scores>self.det_score).view(-1) # 0.05 for pretrain, 0.2 for finetune
                # pos_idxs = torch.nonzero(scores>0.2).view(-1)#75.43
                # pos_idxs = torch.nonzero(scores>0.23).view(-1)#75.43
                box = box[pos_idxs]
                box.add_field("texts", target.get_field("texts"))
                box.add_field("scale", target.get_field("scale"))
                box.add_field("path", target.get_field("path"))
                # box.add_field("y_trues", target.get_field("y_trues"))
                new_boxes.append(box)
                # new_boxes.append(box)
            image_names = [os.path.basename(str(image.get_field("path"))) for image in new_boxes]
            # self.test_visual(images, new_boxes[0].bbox,image_names[0])
            # pos_idxs = torch.nonzero(scores>thresholds).view(-1)
            # results, other = self.head(rec_features,{"retrieval_samples":new_boxes},images)
            results, other = self.head(rec_features,{"retrieval_samples":new_boxes},images,is_words=is_words)
            # results, other = self.head(rec_features,{"retrieval_samples":targets},images,is_words=is_words)
            # self.test_visual(images, new_boxes[0].bbox,other["select_boxes"][0],image_names[0])
            # results, _ = self.head(rec_features,targets,images)
            return results, other

        # preds, _ = self.head(rec_features, boxes)
        


@registry.ONE_STAGE_HEADS.register("align")
def build_align_head(cfg, in_channels):
    return AlignModule(cfg, in_channels, 
                        Matcher(
                        0.8,
                        0.5,
                        allow_low_quality_matches=False)
                    )
if __name__ == '__main__':
    input = torch.rand([4,256,4,15])
    conv = CharCountHead(256, size=(4,15))
    conv(input, targets= torch.tensor([0,15,2,8]))