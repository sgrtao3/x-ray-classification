import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class X_ray_Classifier(nn.Module):
    def __init__(self):
        super(X_ray_Classifier, self).__init__()
        # Defining the convolution layers and the maxpooling structure
        self.c1 = nn.Conv2d(1, 32, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.c4 = nn.Conv2d(32, 64, kernel_size=3)
        self.mp4 = nn.MaxPool2d(kernel_size=2)
        self.f1 = nn.Linear(16384, 1280)
        self.f2 = nn.Linear(1280, 64)
        self.output = nn.Linear(64, 4)
        # Define the activation function
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.c1(x))
        x = self.act1(self.mp1(x))
        x = self.act1(self.c2(x))
        x = self.act1(self.mp2(x))
        x = self.act1(self.c3(x))
        x = self.act1(self.mp3(x))
        x = self.act1(self.c4(x))
        x = self.act1(self.mp4(x))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.act1(self.f1(x))
        x = self.act1(self.f2(x))
        return self.output(x)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class CustomAlexnet(nn.Module):
    def __init__(self):
        super(CustomAlexnet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        # Alter the final fully connected layer to output 4 classes
        self.model.classifier = \
            nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                          nn.Linear(in_features=9216, out_features=4096,
                                    bias=True),
                          nn.ReLU(inplace=True),
                          nn.Dropout(p=0.5, inplace=False),
                          nn.Linear(in_features=4096, out_features=4096,
                                    bias=True),
                          nn.ReLU(inplace=True),
                          nn.Linear(in_features=4096, out_features=4,
                                    bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.model = models.resnet18(pretrained=True)

        self.model.fc = nn.Sequential(
                            nn.Linear(512, 512, bias=True),
                            nn.ReLU(),
                            nn.Linear(512, 4, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class CustomSqueezenet(nn.Module):
    def __init__(self):
        super(CustomSqueezenet, self).__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        # Change final layer for our case
        self.model.classifier[1] = nn.Conv2d(512, 4,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))

    def forward(self, x):
        x = self.model(x)
        return x


class CustomGooglenet(nn.Module):
    def __init__(self):
        super(CustomGooglenet, self).__init__()
        model = models.googlenet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(in_features=1024, out_features=4, bias=True)
        self.model = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1,
                                             padding=1, dilation=1, groups=1,
                                             bias=True), model)

    def forward(self, x):
        x = self.model(x)
        return x


class basic_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# classes needed for the
class X_ray_RESnet(nn.Module):

    def __init__(self, block, layers, num_classes=4):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def xray_RESNET():
    layers = [3, 4, 6, 3]
    model = X_ray_RESnet(basic_block, layers)
    return model
