import torch
from torch import nn
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import os
try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
  font_path = os.path.join('/home/ma-user/modelarts/user-job-dir/RetrievalHuaWei/', 'fonts/simsun.ttc')
except:
  run_on_remote = False
  font_path = 'fonts/simsun.ttc'

# fontpath = 'fonts/simsun.ttc'
# font = ImageFont.truetype(fontpath, 16)
# save_path = 'queryImages'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
def draw_image(text, output_size=(100,32)):
    # font_size
    w = 16 * len(text) + 6*2
    font = ImageFont.truetype(font_path, 16)
    img_bk = Image.fromarray(np.zeros([24,w,3]).astype(np.uint8)+255)
    draw = ImageDraw.Draw(img_bk)
    draw.text((6,4), text, font=font, fill=(0, 0, 0))
    img_bk = img_bk.resize(output_size)
    img_bk = np.array(img_bk)
    # cv2.imwrite(os.path.join(save_path,'{}.jpg'.format(text)), img_bk)
    return img_bk
class TextFeat(nn.Module):
    def __init__(self, outdim=256, nc=3, leakyRelu=False):
        super(TextFeat, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 128, 256, 256, outdim, outdim]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('textfeatconv_{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
    def normalize(self, data):
        return data/255.0
    def forward(self, texts):
        inputs = [torch.tensor(draw_image(text, output_size=(264, 32))) for text in texts]
        inputs = torch.stack(inputs).permute(0,3,1,2).type_as(list(self.parameters())[0])
        inputs = self.normalize(inputs)
        # conv features
        conv = self.cnn(inputs)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2).permute(0,2,1).contiguous()
        return conv
# data =  draw_image("华中科", output_size=(280, 32))
# cnn = TextFeat()
# out = cnn(["华中科技大学","清华大学"])
# cv2.imwrite('save.jpg', draw_image("华中科", output_size=(280, 32)))20.
