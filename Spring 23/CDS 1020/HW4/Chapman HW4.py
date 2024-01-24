import torch

### PART 1 ###
# Q1
x = torch.randn(())
y = torch.randn(())

# Q2
z = torch.where(x > y, x + y, x - y)

# Q3
X = torch.randn((5, 4))
print(X.shape)

