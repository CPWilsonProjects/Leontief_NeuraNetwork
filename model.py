import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self, n_region, n_industry):
        super().__init__()
        self.n_region = n_region
        self.n_industry = n_industry
        self.input_size = self.n_region*self.n_region*self.n_industry
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.input_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = logits.reshape(self.n_region, self.n_region*self.n_industry)
        trade_optim = logits.softmax(0)
        return trade_optim

class GetDemand(nn.Module):
    def __init__(self, tech_mat, final_desire, n_region):
        super().__init__()
        self.tech_mat = tech_mat.detach().clone()
        self.final_desire = final_desire.detach().clone()
        self.n_region = n_region
        self.n_industry = int(self.tech_mat.shape[0]/self.n_region)
        self.trade_indices = torch.empty(self.n_region, self.n_region*self.n_industry, dtype=torch.int64)

        for ii in range(self.n_industry):
            row_indices = self.n_industry*torch.arange(0, self.n_region, dtype=self.trade_indices.dtype) + ii
            self.trade_indices[:, row_indices] = row_indices.unsqueeze(1)

            # for jj in range(self.n_region):
                # col_values = torch.cat([row_indices[0:jj], row_indices[jj+1:]])
                # self.trade_indices[:, row_indices[jj]] = col_values

    def forward(self, trade_coeffs):
        trade_mat = torch.zeros_like(self.tech_mat).scatter_(0, self.trade_indices, trade_coeffs)
        techtrade_mat = torch.matmul(trade_mat, self.tech_mat)
        final_demand = torch.matmul(trade_mat, self.final_desire)
        total_demand = torch.linalg.solve(torch.eye(trade_mat.shape[0]) - techtrade_mat, final_demand)
        return total_demand

