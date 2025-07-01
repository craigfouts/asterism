from torch import nn, optim

NORM = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACT = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}
OPTIM = {'adam': optim.Adam, 'sgd': optim.SGD}

class MLP(nn.Sequential):
    def __init__(self, *channels, bias=True, norm_layer=None, act_layer=None, dropout=0., final_bias=True, final_norm=None, final_act=None, final_dropout=0.):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, norm_layer, act_layer, dropout))

        modules.append(self.layer(channels[-2%len(channels)], channels[-1], final_bias, final_norm, final_act, final_dropout))

        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, norm_layer=None, act_layer=None, dropout=0.):
        if out_channels is None:
            out_channels = in_channels

        module = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if norm_layer is not None:
            module.append(NORM[norm_layer](out_channels))

        if act_layer is not None:
            module.append(ACT[act_layer]())

        if dropout > 0.:
            module.append(nn.Dropout(dropout))

        return module
    