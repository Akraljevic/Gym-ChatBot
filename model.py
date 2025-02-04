import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.3)  

        self.l2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)  
        
        self.l3 = nn.Linear(hidden_size, hidden_size // 2)  
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(0.3)  

        self.l4 = nn.Linear(hidden_size // 2, num_classes)  

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=0.01) 
        out = self.dropout1(out)  

        out = self.l2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, negative_slope=0.01) 
        out = self.dropout2(out)  

        out = self.l3(out)
        out = self.bn3(out)
        out = F.leaky_relu(out, negative_slope=0.01) 
        out = self.dropout3(out)  

        out = self.l4(out)
        return out
