import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
class AttentionUnit(nn.Module):
  def __init__(self, sDim, xDim, attDim):
    super(AttentionUnit, self).__init__()

    self.sDim = sDim
    self.xDim = xDim
    self.attDim = attDim

    self.sEmbed = nn.Linear(sDim, attDim)
    self.xEmbed = nn.Linear(xDim, attDim)
    self.wEmbed = nn.Linear(attDim, 1)

    # self.init_weights()

  def init_weights(self):
    init.normal_(self.sEmbed.weight, std=0.01)
    init.constant_(self.sEmbed.bias, 0)
    init.normal_(self.xEmbed.weight, std=0.01)
    init.constant_(self.xEmbed.bias, 0)
    init.normal_(self.wEmbed.weight, std=0.01)
    init.constant_(self.wEmbed.bias, 0)

  def forward(self, x, sPrev):
    batch_size, T, _ = x.size()                      # [b x T x xDim]
    x = x.view(-1, self.xDim)                        # [(b x T) x xDim]
    xProj = self.xEmbed(x)                           # [(b x T) x attDim]
    xProj = xProj.view(batch_size, T, -1)            # [b x T x attDim]

    sPrev = sPrev.squeeze(0)
    sProj = self.sEmbed(sPrev)                       # [b x attDim]
    sProj = torch.unsqueeze(sProj, 1)                # [b x 1 x attDim]
    sProj = sProj.expand(batch_size, T, self.attDim) # [b x T x attDim]

    sumTanh = torch.tanh(sProj + xProj)
    sumTanh = sumTanh.view(-1, self.attDim)

    vProj = self.wEmbed(sumTanh) # [(b x T) x 1]
    vProj = vProj.view(batch_size, T)

    alpha = F.softmax(vProj, dim=1) # attention weights for each sample in the minibatch

    return alpha


class DecoderUnit(nn.Module):
  def __init__(self, sDim, xDim, yDim, attDim):
    super(DecoderUnit, self).__init__()
    self.sDim = sDim
    self.xDim = xDim
    self.yDim = yDim
    self.attDim = attDim
    self.emdDim = attDim

    self.attention_unit = AttentionUnit(sDim, xDim, attDim)
    self.tgt_embedding = nn.Embedding(yDim+1, self.emdDim) # the last is used for <BOS> 
    self.gru = nn.GRU(input_size=xDim+self.emdDim, hidden_size=sDim, batch_first=True)
    self.fclayer = nn.Linear(sDim, sDim)

    # self.init_weights()

#   def init_weights(self):
#     init.normal_(self.tgt_embedding.weight, std=0.01)
#     init.normal_(self.fc.weight, std=0.01)
#     init.constant_(self.fc.bias, 0)

  def forward(self, x, sPrev, yPrev):
    # x: feature sequence from the image decoder.
    batch_size, T, _ = x.size()
    alpha = self.attention_unit(x, sPrev)
    context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
    yProj = self.tgt_embedding(yPrev.long())
    # self.gru.flatten_parameters()
    output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), sPrev)
    output = output.squeeze(1)

    output = self.fclayer(output)
    return output, state
class AttentionRecognitionHead(nn.Module):
  """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """
  def __init__(self, seg_num, in_planes, attDim, max_len_labels):
    super(AttentionRecognitionHead, self).__init__()
    # self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
    self.seg_num = nn.Embedding(seg_num, attDim)
    self.in_planes = in_planes
    self.max_len_labels = max_len_labels

    self.decodermodule = DecoderUnit(sDim=attDim, xDim=attDim, yDim=max_len_labels, attDim=attDim)
    conv_func = conv_with_kaiming_uniform(True, True, use_deformable=False, use_bn=False)
    self.conv = nn.Sequential(
            conv_func(in_planes, in_planes, 3, stride=(2, 1)),
            conv_func(in_planes, attDim, 3, stride=(2, 1))
        )

  def forward(self, x, seg_num):
    x = self.conv(x).squeeze(2).permute(0,2,1).contiguous()
    batch_size = x.size(0)
    T = x.size(1)
    # Decoder
    seg_num = torch.tensor([seg_num]*batch_size).long().to(x.device)
    state = self.seg_num(seg_num)[None]
    outputs = []

    for i in range(T):
    #   if i == 0:
    #     y_prev = torch.zeros((batch_size)).fill_(self.num_classes) # the last one is used as the <BOS>.
    #   else:
    #     y_prev = targets[:,i-1]
      y_prev = torch.zeros((batch_size)).fill_(i).to(x.device)
      output, state = self.decodermodule(x, state, y_prev)
      outputs.append(output)
    outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    return outputs
if __name__ == '__main__':
    au = AttentionRecognitionHead(4,128,128,128,20)
    
    x = torch.randn([8, 128, 4, 15])
    s = torch.tensor([0,1,2,3,2,2,3,0])

    out = au(x, 0)
    print(out.shape)