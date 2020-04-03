# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
from search.modules.layers import *
import json
import torch.nn as nn


# build the proxylessNAS network from a JSON file
def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class DenseBlock(MyModule):

    def __init__(self, conv, shortcut):
        super(DenseBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv.is_zero_layer():
            res = self.conv(x)
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.conv(x)
        else:
            conv_x = self.conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s)' % (
            self.conv.module_str
        )

    @property
    def config(self):
        return {
            'name': DenseBlock.__name__,
            'conv': self.conv.config,
        }

    @staticmethod
    def build_from_config(config):
        conv = set_layer_from_config(config['conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return DenseBlock(conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class ResidualBlock(MyModule):

    def __init__(self, conv, shortcut, out_channel):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut
        self.out_channel = out_channel
        self.bn_1 = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        # forward through the MixedEdge class i.e. the forward function
        # it takes care of everything related to different operations
        if self.conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.conv(x)
        else:
            conv_x = self.conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        res = self.bn_1(res)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': ResidualBlock.__name__,
            'conv': self.conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv = set_layer_from_config(config['conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return ResidualBlock(conv, shortcut, config['out_channel'])

    def get_flops(self, x):
        flops1, conv_x = self.conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        # forward through the MixedEdge class i.e. the forward function
        # it takes care of everything related to different operations
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x


class SuperNets(MyNetwork):

    def __init__(self, fc_0, blocks, first_conv_ch):
        super(SuperNets, self).__init__()

        self.fc_0 = fc_0
        self.blocks = nn.ModuleList(blocks)
        self.first_conv_ch = first_conv_ch
        self.conv_0 = ConvLayer(first_conv_ch, 1, 3)

    def forward(self, x):
        x = self.fc_0(x)
        x = x.view(-1, self.first_conv_ch, 2, 2)
        for block in self.blocks:
            x = block(x)
            # print("Tensor Shape: ", str(x.shape) + str(block))
        x = self.conv_0(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': SuperNets.__name__,
            'bn': self.get_bn_param(),
            'first_linear': self.fc_0.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_linear': self.fc_1.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        first_conv_ch = set_layer_from_config(config['first_conv_ch'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = SuperNets(first_conv, blocks, first_conv_ch)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.fc_0.get_flops(x)
        x = x.view(-1, self.first_conv_ch, 2, 2)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.conv_0.get_flops(x)
        flop += delta_flop

        return flop, x
