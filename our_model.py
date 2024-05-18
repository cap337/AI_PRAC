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

class DeepNetworkWithModerateLayers(nn.Module):
    def __init__(self):
        super(DeepNetworkWithModerateLayers, self).__init__()
        self.input_layer = nn.Linear(10, 64)
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.dropout(F.relu(self.layer3(x)))
        x = self.dropout(F.relu(self.layer4(x)))
        x = self.output_layer(x)
        return x


class ShallowNetworkWithWideLayers(nn.Module):
    def __init__(self):
        super(ShallowNetworkWithWideLayers, self).__init__()
        self.input_layer = nn.Linear(10, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.output_layer(x)
        return x


class DeepNetworkWithBatchNormalization(nn.Module):
    def __init__(self):
        super(DeepNetworkWithBatchNormalization, self).__init__()
        self.input_layer = nn.Linear(10, 64)
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.bn1(F.relu(self.input_layer(x)))
        x = self.dropout(self.bn2(F.relu(self.layer1(x))))
        x = self.dropout(self.bn3(F.relu(self.layer2(x))))
        x = self.dropout(self.bn4(F.relu(self.layer3(x))))
        x = self.bn5(F.relu(self.layer4(x)))
        x = self.output_layer(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(F.relu(out))
        return out + identity

class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()
        self.input_layer = nn.Linear(10, 64)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 64)
        self.output_layer = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(self.res1(x))
        x = self.dropout(self.res2(x))
        x = self.dropout(self.res3(x))
        x = self.output_layer(x)
        return x


class WideNetworkWithDropout(nn.Module):
    def __init__(self):
        super(WideNetworkWithDropout, self).__init__()
        self.input_layer = nn.Linear(10, 256)
        self.layer1 = nn.Linear(256, 512)
        self.layer2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.output_layer(x)
        return x
