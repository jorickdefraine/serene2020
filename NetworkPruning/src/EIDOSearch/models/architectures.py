import torch.nn.functional as F
from torch import nn
from torchvision import models as tv_models


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=True)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, bias=True)
        self.r2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10, bias=True)

    def forward(self, img):
        output = self.conv1(img)
        output = self.r1(output)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = self.r2(output)
        output = F.max_pool2d(output, 2)
        output = output.view(img.size(0), -1)
        output = self.fc1(output)
        output = self.r3(output)
        output = self.fc2(output)

        return output


class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300, bias=True)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        for lay in self.layers:
            out = lay(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Increses dimension via padding, performs identity operations
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option="A"):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(64, num_classes)

    # self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        pool_size = int(out.size(3))
        out = F.avg_pool2d(out, pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(option="A"):
    return ResNet(BasicBlock, [3, 3, 3], option=option)


def resnet32(option="A"):
    return ResNet(BasicBlock, [5, 5, 5], option=option)


def resnet44(option="A"):
    return ResNet(BasicBlock, [7, 7, 7], option=option)


def resnet56(option="A"):
    return ResNet(BasicBlock, [9, 9, 9], option=option)


def resnet110(option="A"):
    return ResNet(BasicBlock, [18, 18, 18], option=option)


def resnet1202(option="A"):
    return ResNet(BasicBlock, [200, 200, 200], option=option)


def resnet101():
    return tv_models.resnet101(False)
