from architecture_stat import architecture_stat
import torch
import sys
from torch import nn
 
# get argument list using sys module
sys.argv


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
            nn.Linear(in_features=100, out_features=18),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    
class LeNet300(nn.Module):

    def __init__(self):
        super(LeNet300, self).__init__()
        
        self.feature=nn.Sequential(
            nn.Linear(32*32,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,18)
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.feature(x)
        probs = F.log_softmax(x, dim=1)
        return x

#model = LeNet5_8_16()
model = LeNet300()
model.load_state_dict(torch.load('/home/paf2020/NetworkPruning/src/logs/'+str(sys.argv[1])+'/models/'+str(sys.argv[2])+'.pt'))
print(architecture_stat(model))

