import torch.nn as nn
import torch.nn.functional as F

class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.input_layer = nn.Linear(10,30)
        self.layer1 = nn.Linear(30, 60)
        self.layer2 = nn.Linear(60, 120)
        self.layer3 = nn.Linear(120, 60)
        self.layer4 = nn.Linear(60, 30)
        self.layer5 = nn.Linear(30, 10)
        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(80)
        self.bn3 = nn.BatchNorm1d(90)
        self.bn4 = nn.BatchNorm1d(62)

        self.output_layer = nn.Linear(10, 3)
        self.dropout = nn.Dropout(p=0.005)  # Dropout rate
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.dropout(F.relu(self.layer3(x)))
        x = self.dropout(F.relu(self.layer4(x)))
        x = self.dropout(F.relu(self.layer5(x)))
        x = self.output_layer(x)
        return x

