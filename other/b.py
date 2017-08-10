import torch
a = torch.randn(8,9)
print(a)
a = a.view(3,-1)

print(a)

print('==============================')
a = torch.randn(8, 10)
print(a)

print(torch.max(a,1))