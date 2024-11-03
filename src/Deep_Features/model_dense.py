import torch.nn as nn

def build_activation(activation_name):
    activation_funcs = nn.ModuleDict([
        ['elu',         nn.ELU()],
        ['log_sigmoid', nn.LogSigmoid()],
        ['sigmoid',     nn.Sigmoid()],
        ['relu',        nn.ReLU()]
    ])
    return activation_funcs[activation_name]

class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout=0.0):
        super(DenseBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.op = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
            build_activation(activation)
        )

    def forward(self, x):
        x = self.op(x)
        return x

class DenseNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation='relu', dropout=0.0):
        super(DenseNetwork, self).__init__()
        self.in_dim = in_dim
        self.activation = build_activation(activation)
        if len(hidden_dims) > 1:
            self.forward_op = nn.Sequential(
                DenseBlock(in_dim, hidden_dims[0], activation, dropout),
                *[DenseBlock(input_dim, output_dim, activation, dropout) for (input_dim, output_dim) in zip(hidden_dims[:-1],hidden_dims[1:])],
                DenseBlock(hidden_dims[-1], out_dim, activation, dropout)
            )
        elif len(hidden_dims) == 1:
            self.forward_op = nn.Sequential(
                DenseBlock(in_dim, hidden_dims[0], activation, dropout),
                DenseBlock(hidden_dims[0], out_dim, activation, dropout)
            )
        else:
            self.forward_op = DenseBlock(in_dim, out_dim, activation, dropout)

    def forward(self, x):
        x = self.forward_op(x.reshape((-1, self.in_dim)))
        return x

def generate_hidden_dims(size, in_dim):
    if size == 1:
        return []

    expansion = {
        2:  [1.5],
        6:  [1.5, 2.0, 1.5, 0.5],
        10: [1.5, 2.5, 5.0, 5.5, 3.5, 2.5, 1.5, 0.5],
        17: [1.5, 2.5, 5.0, 8.0, 10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 5.0, 3.5, 2.5, 1.5, 0.5],
        23: [1.5, 2.5, 5.0, 8.0, 10.0, 15.0, 20.0, 20.0, 18.0, 15.0, 15.0, 12.0, 10.0, 10.0, 8.5, 5.5, 3.0, 3.0, 2.0, 1.0, 0.5],

    }
    assert size in expansion.keys(), f"Unsupported dense network size: {size}."
    return [int(ex*in_dim) for ex in expansion[size]]

def generate_small_hidden_dims(size, in_dim):
    if size == 1:
        return []

    expansion = {
        2:  [0.5],
        6:  [1.0, 0.8, 0.5, 0.2],
        10: [1.0, 1.0, 1.0, 0.8, 0.7, 0.5, 0.3, 0.2],
        17: [1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.2, 1.2, 1.0, 1.0, 0.8, 0.5, 0.3, 0.2],
        23: [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.8, 1.5, 1.2, 1.2, 1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2],

    }
    assert size in expansion.keys(), f"Unsupported dense network size: {size}."
    return [int(ex*in_dim) for ex in expansion[size]]

def dense_predictor(in_dim, out_dim, size, activation='relu', dropout=0.0):
    """
    Build dense NN model with large hidden dimension.
    """
    assert activation in ['elu', 'log_sigmoid', 'sigmoid', 'relu']
    return DenseNetwork(in_dim, out_dim, generate_hidden_dims(size, in_dim), activation=activation, dropout=dropout)

def flat_dense_predictor(in_dim, out_dim, size, activation='relu', dropout=0.0):
    """
    Build dense NN model with small hidden dimension.
    """
    assert activation in ['elu', 'log_sigmoid', 'sigmoid', 'relu']
    return DenseNetwork(in_dim, out_dim, generate_small_hidden_dims(size, in_dim), activation=activation, dropout=dropout)
