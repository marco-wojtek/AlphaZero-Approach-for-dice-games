import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

print(torch.__version__)
#run with python -u "d:\Informatikstudium\Bachelor-Arbeit\Python_code\NN.py" or pyton -u NN.py
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(65, 42),
#             nn.ReLU(),
#             nn.Linear(42, 20),
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
    
# model = NeuralNetwork().to(device)
# print(model)