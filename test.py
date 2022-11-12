import torch
import attentions
import commons

encoder = attentions.Encoder(
      256,
      768,
      2,
      6,
      3,
      0.1)

x = torch.zeros([12,  256, 21])
x_mask = torch.unsqueeze(commons.sequence_mask(torch.ones([12]), x.size(2)), 1).to(x.dtype)
# print(x*x_mask)
print(encoder(x,x_mask).shape)