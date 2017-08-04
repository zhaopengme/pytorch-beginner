import  torch

x = torch.rand(3,3)

print(x)

print(torch.max(x,1))# max(data,dim) dim 取值是从-2到1,取某个维度的最大值
print(torch.max(x,0))