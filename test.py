import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.transpose(0, 1)
y1 = y.contiguous()

print(y)
print(y1)
print(y==y1)
print(y1.is_contiguous())