import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .model import NeuralNetwork
from .model import GetDemand

n_region = 2
n_industry = 3

cost_per_unit_production = torch.tensor([[0, 2, 1],
                                         [1, 0, 1],
                                         [1, 1, 0]], dtype=torch.float32)

base_yield_per_unit_production = torch.tensor([5, 3, 3], dtype=torch.float32)

base_tech_mat = cost_per_unit_production/base_yield_per_unit_production

advantage = torch.tensor([[1, 0, 0], 
                          [0, 1, 0]], dtype=torch.float32)

yield_per_unit_production = (base_yield_per_unit_production + advantage).reshape(n_region*n_industry)
tech_mat = torch.kron(torch.eye(n_region), cost_per_unit_production)/yield_per_unit_production

final_desire = torch.tensor([4, 2, 3, 6, 2, 4], dtype=torch.float32)

gd_obj = GetDemand(tech_mat, final_desire, n_region)
nn_obj = NeuralNetwork(n_region, n_industry)

trade_coeffs = torch.ones(1, n_region, n_region*n_industry, dtype=torch.float32, requires_grad=True)/2

epochs = 1000
learning_rate = 1E-2
optimiser = torch.optim.SGD(nn_obj.parameters(), learning_rate)
loss_tracker = torch.zeros(epochs, dtype=torch.float32)

for ii in range(epochs):
    trade_coeffs = trade_coeffs.detach().requires_grad_(True)
    trade_coeffs_opt = nn_obj(trade_coeffs)
    final_demand_opt = gd_obj(trade_coeffs_opt)
    loss = sum(final_demand_opt)
    loss_tracker[ii] = loss

    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

print(loss_tracker)