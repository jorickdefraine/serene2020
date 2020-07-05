import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 5
IMG_SIZE = 32
N_CLASSES = 18

class LeNet5_8_16(nn.Module):

    def __init__(self):
        super(LeNet5_8_16, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=5*5*16, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=N_CLASSES),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class LeNet5_16_32(nn.Module):

    def __init__(self):
        super(LeNet5_16_32, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=5*5*32, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=N_CLASSES),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class LeNet300_32(nn.Module):

    def __init__(self):
        super(LeNet300_32, self).__init__()
        
        self.feature=nn.Sequential(
            nn.Linear(32*32,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,N_CLASSES)
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.feature(x)
        return x