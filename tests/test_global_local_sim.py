import torch
from torch import nn
class GlobalLocalSimilarity(nn.Module):
    
    def __init__(self,divided_nums = [1,3,5]):
        super(GlobalLocalSimilarity, self).__init__()
        self.divided_nums = divided_nums
        self.normalize = nn.functional.normalize
    def compute_similarity(self,x,y,divided_num=1):
        x = x.view(x.size(0),divided_num,-1)
        y = y.view(y.size(0),divided_num,-1)
        sims = torch.stack([self.normalize(x[:,i,:]).mm(self.normalize(y[:,i,:]).t()) for i in range(divided_num)],dim=-1)
        return sims.mean(dim=-1)
        
    def forward(self, x,y):
        x_tanh = x.tanh()
        y_tanh = y.tanh()
        sims = torch.stack([self.compute_similarity(x_tanh, y_tanh, divided_num) for divided_num in self.divided_nums],dim=-1)
        return sims.mean(dim=-1)
net = GlobalLocalSimilarity()
x = torch.randn([4,15,256])
y = torch.randn([4,15,256])
out = net(x,y)
print(out.shape)