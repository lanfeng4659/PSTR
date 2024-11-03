import editdistance
import argparse
import time
import torch
from torch import nn
x = torch.randn([4,15,128])
y = torch.randn([3,15,128])
# class DynamicMaxSimilarity(nn.Module):
    
#     def __init__(self,frame_num):
#         super(DynamicMaxSimilarity, self).__init__()
#         self.frame_num = frame_num
#     def sim(self,x,y):
#         # print(x.shape)
#         x_nor = torch.nn.functional.normalize(x.view(1,-1).tanh())
#         y_nor = torch.nn.functional.normalize(y.view(1,-1).tanh())
#         return x_nor.mm(y_nor.t())
#     def push_similarity(self,global_sim, local_sim, steps):
#         return (global_sim*(steps-1)+local_sim)/steps
#     def forward(self,a,b):
#         si = torch.zeros([self.frame_num+1, self.frame_num+1])
#         for i in range(1, self.frame_num+1):
#             for j in range(1, self.frame_num+1):
#                 local_sim = self.sim(a[i-1],b[j-1])
#                 si[i][j] = max([si[i-1][j]+local_sim, si[i][j-1]+local_sim, si[i-1][j-1]+local_sim])
#                 si[i][j] = max([self.push_similarity(si[i-1][j], local_sim, max(i,j)), 
#                                 self.push_similarity(si[i][j-1], local_sim, max(i,j)), 
#                                 self.push_similarity(si[i-1][j-1], local_sim, max(i,j))])
#                 # print(i,j,local_sim, si[i-1][j],si[i][j-1],si[i-1][j-1])
#         print(torch.diagonal(si))
#         return si[-1][-1]
class DynamicMaxSimilarity(nn.Module):
    
    def __init__(self,frame_num):
        super(DynamicMaxSimilarity, self).__init__()
        self.frame_num = frame_num
    # def sim(self,x,y):
    #     # print(x.shape)
    #     x_nor = torch.nn.functional.normalize(x.view(x.size(0),-1).tanh())
    #     y_nor = torch.nn.functional.normalize(y.view(y.size(0),-1).tanh())
    #     return x_nor.mm(y_nor.t())
    def sim(self,x,y):
        x_nor = torch.nn.functional.normalize(x.view(-1,x.size(-1)).tanh()) # x_bw,c
        y_nor = torch.nn.functional.normalize(y.view(-1,y.size(-1)).tanh()) # y_bw,c
        similarity = x_nor.mm(y_nor.t()) # (x_bw,y_bw)
        similarity = similarity.reshape([x.size(0),x.size(1),y.size(0),y.size(1)])
        return similarity.permute(0,2,1,3)
    def push_similarity(self,global_sim, local_sim, steps):
        return (global_sim*(steps-1)+local_sim)/steps
    def forward(self,a,b):
        si = torch.zeros([a.size(0),b.size(0),self.frame_num+1, self.frame_num+1])
        local_similarity = self.sim(a,b)
        for i in range(1, self.frame_num+1):
            for j in range(1, self.frame_num+1):
                local_sim = local_similarity[:,:,i-1,j-1]
                all_sim = torch.stack([self.push_similarity(si[:,:,i-1,j], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i,j-1], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i-1,j-1], local_sim, max(i,j))]
                                       ,dim=-1)
                si[:,:,i,j] = torch.max(all_sim,dim=-1)[0]
        # for i in range(a.size(0)):
        #     for j in range(b.size(0)):
        #         print(torch.diagonal(si[i,j]))
        return si[:,:,-1,-1]
if __name__ == '__main__':
    steps = 5
    dms = DynamicMaxSimilarity(steps)
    x = torch.randn([2,steps,128])
    y = torch.randn([2,steps,128])
    ms = dms(x,x)
    print(ms)
    # print(dms.sim(x,x))
    # torch.mm(x[:,0,:],y[:,0,:].t())
