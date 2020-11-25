"""Universal network struture unit definition."""
import torch
import math
from torch import nn
import torchvision
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

def define_squeeze_unit(basic_channel_size):
    """Define a 1x1 squeeze convolution with norm and activation."""
    conv = nn.Conv2d(2 * basic_channel_size, basic_channel_size, kernel_size=1,
                     stride=1, padding=0, bias=False)
    norm = nn.BatchNorm2d(basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                     stride=1, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_halve_unit(basic_channel_size):
    """Define a 4x4 stride 2 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=4,
                     stride=2, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_depthwise_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv1 = nn.Conv2d(basic_channel_size, 2 * basic_channel_size,
                      kernel_size=1, stride=1, padding=0, bias=False)
    norm1 = nn.BatchNorm2d(2 * basic_channel_size)
    relu1 = nn.LeakyReLU(0.1)
    conv2 = nn.Conv2d(2 * basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                      stride=1, padding=1, bias=False, groups=2 * basic_channel_size)
    norm2 = nn.BatchNorm2d(2 * basic_channel_size)
    relu2 = nn.LeakyReLU(0.1)
    layers = [conv1, norm1, relu1, conv2, norm2, relu2]
    return layers


def define_detector_block(basic_channel_size):
    """Define a unit composite of a squeeze and expand unit."""
    layers = []
    layers += define_squeeze_unit(basic_channel_size)
    layers += define_expand_unit(basic_channel_size)
    return layers

class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""
    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        # 1
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 2
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 3
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 4
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 5
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# vgg backbone

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 1024, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

class ResNet18(nn.Module):
    def __init__(self, block, layers, aux_classes=1000, classes=100, domains=3):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)#resnet 18

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

class ResNet50(nn.Module):
    def __init__(self, block, layers, aux_classes=1000, classes=100, domains=3):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2) #resnet50

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet50(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
