import torch
import torch.nn as nn

class KAMP_CNN(nn.Module):
  def __init__(self, layermode):
    super(KAMP_CNN, self).__init__()
    if layermode >= 0:
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Dropout(p=0.2)) 
        
    if layermode >= 1:
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Dropout(p=0.2))
        
    if layermode >= 2:
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Dropout(p=0.2))

    self.conv4 = nn.Sequential(
        nn.Conv1d(in_channels=100, out_channels=4, kernel_size=2, stride=1, padding='same'),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=1, stride=1))

    self.final_pool = nn.AdaptiveAvgPool1d(1)
    self.linear = nn.Linear(4, 4)
  
  def forward(self, input):
    input = input.unsqueeze(1)
    out = self.conv1(input)
    # out = self.conv2(out)
    # out = self.conv3(out)
    out = self.conv4(out)
    out = self.final_pool(out)
    out = self.linear(out.squeeze(-1))
    return out

# model_check = KAMP_CNN(1)
# print(model_check)