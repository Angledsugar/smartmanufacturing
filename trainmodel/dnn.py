import torch
import torch.nn as nn

class KAMP_DNN(nn.Module):
  def __init__(self):
    super(KAMP_DNN, self).__init__()
    self.layer1 = nn.Linear(in_features = 4, out_features = 100)
    self.layer2 = nn.Linear(in_features = 100, out_features = 100)
    self.layer3 = nn.Linear(in_features = 100, out_features = 100)
    self.layer4 = nn.Linear(in_features = 100, out_features = 4)
    
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()

  def forward(self, input):
    out = self.layer1(input)
    out = self.relu(out)
    out = self.dropout(out)
    
    out = self.layer2(out)
    out = self.relu(out)
    out = self.dropout(out)
    
    out = self.layer3(out)
    out = self.relu(out)
    out = self.dropout(out)
    
    out = self.layer4(out)
    
    return out

# model_check = KAMP_DNN()
# print(model_check)