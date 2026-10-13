import os
import torch
import pyvis
import webbrowser
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .model import NeuralNetwork
from .model import GetDemand

# Setup
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

# Training and optimisation
epochs = 1000
learning_rate = 1E-2
optimiser = torch.optim.SGD(nn_obj.parameters(), learning_rate)
loss_tracker = torch.zeros(epochs, dtype=torch.float32)

for ii in range(epochs):
    trade_coeffs = trade_coeffs.detach().requires_grad_(True)
    trade_coeffs_opt = nn_obj(trade_coeffs)
    total_demand_opt = gd_obj(trade_coeffs_opt)
    loss = sum(total_demand_opt)
    loss_tracker[ii] = loss

    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

# Plotting
techtrade_mat = torch.matmul(gd_obj.trade_mat, gd_obj.tech_mat).detach()
final_demand = torch.matmul(gd_obj.trade_mat, final_desire).detach()
final_desire_mat = final_desire.diag().detach()
final_demand_mat = torch.matmul(gd_obj.trade_mat, final_desire.diag()).detach()
total_demand = torch.linalg.solve(torch.eye(tech_mat.shape[0]) - tech_mat, final_desire).detach()
total_demand_opt = torch.linalg.solve(torch.eye(tech_mat.shape[0]) - techtrade_mat, final_demand).detach()
inter_demand_mat = torch.matmul(tech_mat, total_demand.diag()).detach()
inter_demand_mat_opt = torch.matmul(techtrade_mat, total_demand_opt.diag()).detach()

tech_network = pyvis.network.Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=True)
techtrade_network = pyvis.network.Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=True)
nodes = n_region*['wood', 'mine', 'farm']
colour = n_industry*["#ff005d"] + n_industry*["#00D9FF"]

tech_network.add_nodes(list(range(n_region*n_industry)), label=nodes, size=total_demand.numpy().tolist(), color=colour)
techtrade_network.add_nodes(list(range(n_region*n_industry)), label=nodes, size=total_demand_opt.numpy().tolist(), color=colour)

for ii in range(techtrade_mat.shape[0]):
    for jj in range(techtrade_mat.shape[1]):
        edge = (inter_demand_mat[ii, jj] + final_desire_mat[ii, jj]).item()
        edge_opt = (inter_demand_mat_opt[ii, jj] + final_demand_mat[ii, jj]).item()

        if edge > 1E-3:
            tech_network.add_edge(ii, jj, value=edge)
        if edge_opt > 1E-3:
            techtrade_network.add_edge(ii, jj, value=edge_opt)

tech_network.set_edge_smooth('dynamic')
techtrade_network.set_edge_smooth('dynamic')

tech_network.show('network.html')
techtrade_network.show('network_opt.html')

webbrowser.open("file://" + os.path.abspath("network.html"))
webbrowser.open("file://" + os.path.abspath("network_opt.html"))
stop = 1