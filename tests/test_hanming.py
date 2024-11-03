import torch
a = torch.tensor([[1,0,1],[0,1,1],[1,1,1]]).float()
b = torch.tensor([[1,0,1],[0,0,1]]).float()

c = a.mm(b.t())
d = (a.sum(dim=1)[:,None].repeat((1,b.size(0))) + b.sum(dim=1)[None,:].repeat((a.size(0),1)))
print(a)
print(b)
print(2*c/d)