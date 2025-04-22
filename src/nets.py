import torch.nn as nn

ACTIVATION = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}

def layer(input_dim, output_dim, bias=True, batch_norm=False, activation='relu', dropout=0., **kwargs):
    """Constructs a neural network layer with optional batch normalization, 
    activation, and dropout.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    output_dim : int
        Output dimensionality.
    bias : bool, default=True
        Whether to include a bias term.
    batch_norm : bool, default=False
        Whether to include batch normalization.
    activation : str, default='relu'
        Activation function.
    dropout : float, default=0.0
        Amount of dropout.

    Yields
    ------
    Module
        Neural network layer components.
    """
    
    yield nn.Linear(input_dim, output_dim, bias=bias)

    if batch_norm:
        yield nn.BatchNorm1d(output_dim)

    if activation is not None:
        yield ACTIVATION[activation](**kwargs)

    if dropout > 0.:
        yield nn.Dropout(dropout)

def mlp(layers, bias=True, final_bias=True, batch_norm=False, final_norm=False, activation='relu', final_act=None, dropout=0., final_drop=0.):
    """Constructs a multilayer perceptron with optional batch normalizations,
    activations, and dropouts.
    
    Parameters
    ----------
    layers : tuple | list
        Layer dimensionalities.
    bias : bool, default=True
        Whether to include hidden bias terms.
    final_bias : bool, default=True
        Whether to include a final bias term.
    activation : str, default='relu'
        Hidden activation function.
    final_act : str, default=None
        Final activation function.
    batch_norm : bool, default=False
        Whether to include hidden batch normalization.
    final_norm : bool, default=False
        Whether to include final batch normalization.
    dropout : float, default=0.0
        Amount of hidden dropout.
    final_drop : float, default=0.0
        Amount of final dropout.

    Yields
    ------
    Module
        Neural network layers.
    """
    
    n_layers = len(layers)

    for i in range(1, n_layers):
        if i < n_layers - 1:
            yield from layer(layers[i - 1], layers[i], bias, batch_norm, activation, dropout)
        else:
            yield from layer(layers[i - 1], layers[i], final_bias, final_norm, final_act, final_drop)

class MLP(nn.Module):
    """Implementation of a multilayer perceptron.
    
    Parameters
    ----------
    layers : tuple | list
        Layer dimensionalities.
    bias : bool, default=True
        Whether to include hidden bias terms.
    final_bias : bool, default=True
        Whether to include a final bias term.
    activation : str, default='relu'
        Hidden activation function.
    final_act : str, default=None
        Final activation function.
    batch_norm : bool, default=False
        Whether to include hidden batch normalization.
    final_norm : bool, default=False
        Whether to include final batch normalization.
    dropout : float, default=0.0
        Amount of hidden dropout.
    final_drop : float, default=0.0
        Amount of final dropout.

    Attributes
    ----------
    net : Sequential
        Neural network model.

    Usage
    -----
    >>> layers = (100, 250, 100)
    >>> mlp = MLP(layers, *args, **kwargs)
    >>> output = mlp(input)
    """
    
    def __init__(self, layers, bias=True, final_bias=True, batch_norm=False, final_norm=False, activation='relu', final_act=None, dropout=0., final_drop=0.):
        super().__init__()

        self.layers = layers
        self.bias = bias
        self.final_bias = final_bias
        self.batch_norm = batch_norm
        self.final_norm = final_norm
        self.activation = activation
        self.final_act = final_act
        self.dropout = dropout
        self.final_drop = final_drop

        self.net = nn.Sequential(*list(mlp(layers, bias, final_bias, batch_norm, final_norm, activation, final_act, dropout, final_drop)))

    def forward(self, x):
        """Performs a single forward pass of the network.
        
        Parameters
        ----------
        x : Tensor
            Network input.

        Returns
        -------
        Tensor
            Network output.
        """
                
        return self.net(x)
