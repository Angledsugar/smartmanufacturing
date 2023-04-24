import torch
import torch.nn as nn

class KAMP_RNN(nn.Module):
  def __init__(self):
    super(KAMP_RNN, self).__init__()
    self.lstm = nn.LSTM(input_size =4, hidden_size =100, num_layers =2,
                  batch_first=True, dropout =0.2)
    self.fc = nn.Linear(in_features =100, out_features =4)

  def forward(self, input):
    input = input.unsqueeze(1)
    out, _ = self.lstm(input)
    out = out.view(-1,100)
    output =self.fc(out)
    return output
 
# model_check = KAMP_RNN()
# print(model_check)