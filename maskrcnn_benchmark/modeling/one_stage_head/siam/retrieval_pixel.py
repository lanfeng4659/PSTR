import torch
from torch import nn
from torch.nn import functional as F
import cv2
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos
from maskrcnn_benchmark.modeling.rpn.east.east import build_east
from maskrcnn_benchmark.modeling.rpn.fast_center.fast_center import build_fast_center
from maskrcnn_benchmark.utils.text_util import TextGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, cat_boxlist_texts
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .core import xcorr_slow, xcorr_fast, xcorr_depthwise
from torch.autograd import Variable
import string
import random
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers import SigmoidFocalLoss
import torch.nn.functional as F
INF = 10000000000
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

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

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
class WordEmbedding(nn.Module):
    def __init__(self,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      max_length=10,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True):
        super(WordEmbedding, self).__init__()
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_embedding = nn.Embedding(len(self.lexicon), embedding_dim)
        self.char_encoder = nn.Sequential(
            nn.Linear(embedding_dim, char_vector_dim),
            nn.ReLU(inplace=True)
        )
        # self.rnn = nn.LSTM(char_vector_dim, out_channels,num_layers=1,bidirectional=bidirectional)
        self.rnn = BidirectionalLSTM(char_vector_dim, 256, out_channels)
    def forward(self,inputs):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        embeddings_batch = []
        for word in inputs:
            embeddings = self.char_embedding(word)
            embeddings_batch.append(
                nn.functional.interpolate(
                    embeddings[None,None,...], 
                    size=(self.max_length,self.embedding_dim), 
                    mode='bilinear', 
                    align_corners=True)
            )
        embeddings_batch = torch.cat(embeddings_batch,dim=1)[0] # [b, self.max_length, embedding_dim]
        char_vector = self.char_encoder(embeddings_batch)
        char_vector = char_vector.permute(1, 0, 2).contiguous() # [w, b, c]
        x = self.rnn(char_vector)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        return x

class SiamHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SiamHead, self).__init__()
        
        out_channels = 128
        in_channels = 256
        self.text_generator = TextGenerator()
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=15,
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
        # self.
        # self.embedding_nor = nn.Linear(out_channels * 2, 32)
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
    def compute_loss(self, embedding1, embedding2, words1,words2):
        # print(embedding1)
        iou = self.compute_similarity(embedding1, embedding2)
        # print(iou)
        similarity = self.text_generator.calculate_similarity_matric(words1, words2)
        loss = self.sim_loss_func(iou, torch.tensor(similarity).type_as(iou))
        # print(loss)
        return loss.max(dim=1)[0].mean()
    def compute_similarity(self,embedding1, embedding2):
        # print(embedding1.shape,embedding2.shape)
        embedding1_nor = nn.functional.normalize(embedding1.tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize(embedding2.tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        # print(inter.shape)
        # print(inter)
        return inter
    def xcorr_slow(self, words, features):
        words = F.normalize(words.view(words.size(0),-1,1,1))
        features = F.normalize(features,dim=1)
        # print(words.shape)
        similarity = F.conv2d(features, words)
    def forward(self, x, proposals,images=None):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        for _ in x:
            print(_.shape)
        if self.training:
            texts = []
            new_proposals = []
            for proposals_per_im in proposals:
                idxs, new_texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                # texts.extend(new_texts)
                new_proposals.append(proposals_per_im[idxs])
            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(rois.device) for text in texts]
            words_embedding = self.word_embedding(words)
            # words_embedding_nor = nn.functional.normalize(words_embedding.tanh()).view(words_embedding.size(0),-1)
            wi_loss = self.compute_loss(words_embedding.detach(), imgs_embedding, texts, texts)
            ww_loss = self.compute_loss(words_embedding, words_embedding, texts, texts)
            ii_loss = self.compute_loss(imgs_embedding, imgs_embedding, texts, texts)
            loss = {"loss_wi":wi_loss*10,"loss_ww":ww_loss*10,"loss_ii":ii_loss*10}
            return None,loss
        else:
            for proposals_per_im in proposals:
                idxs, texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                if len(texts) == 0:
                    words_embedding_nor = torch.zeros([0,1920]).type_as(x[0])
                else:
                    words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x[0].device) for text in texts]
                    words_embedding = self.word_embedding(words)
                    words_embedding_nor = nn.functional.normalize(words_embedding.tanh().view(words_embedding.size(0),-1))
                if proposals_per_im.bbox.size(0) == 0:
                    imgs_embedding_nor = torch.zeros([0,1920]).type_as(x[0])
                else:
                    rois = self.pooler(x, [proposals_per_im])
                    imgs_embedding = self.image_embedding(rois)
                    imgs_embedding_nor = nn.functional.normalize(imgs_embedding.tanh().view(imgs_embedding.size(0),-1))
                proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
                proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)

            return proposals, {}