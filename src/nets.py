import torch.nn as nn

NORMALIZATION = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACTIVATION = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}

class MLP(nn.Sequential):
    """Implementation of a multilayer perceptron
    
    Parameters
    ----------
    channels : int
        Number of channels in each layer.
    bias : bool, defaul=True
        Whether to include additive bias in each hidden layer.
    normalization : str, default=None
        Hidden normalization function.
    activation : str, default=None
        Hidden activation function.
    dropout : float, default=0.0
        Amount of dropout in each hidden layer.
    out_bias : bool, default=True
        Whether to include additive bias in the output layer.
    out_normalization : str, default=None
        Output normalization function.
    out_activation : str, default=None
        Output activation function.
    out_dropout : float, default=0.0
        Amount of dropout in the output layer.

    Attributes
    ----------
    None

    Usage
    -----
    >>> model = MLP(in_channels, hidden_channels, ..., out_channels, **kwargs)
    >>> output = model(data)
    """

    def __init__(self, *channels, bias=True, normalization=None, activation=None, dropout=0., out_bias=True, out_normalization=None, out_activation=None, out_dropout=0.):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, normalization, activation, dropout))

        modules.append(self.layer(channels[-2%len(channels)], channels[-1], out_bias, out_normalization, out_activation, out_dropout))

        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, normalization=None, activation=None, dropout=0.):
        """Constructs a single neural network layer with optional normalization,
        activation, and dropout modules.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int, default=None
            Number of output channels.
        bias : bool, default=True
            Whether to include additive bias.
        normalization : str, default=None
            Normalization function.
        activation : str, default=None
            Activation function.
        dropout : float, default=0.0
            Amount of dropout
        
        Returns
        -------
        Sequential
            Neural network layer module.
        """

        if out_channels is None:
            out_channels = in_channels

        module = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if normalization is not None:
            module.append(NORMALIZATION[normalization](out_channels))

        if activation is not None:
            module.append(ACTIVATION[activation]())

        if dropout > 0.:
            module.append(nn.Dropout(dropout))

        return module
