from torch import nn

from fosco.common.activations import activation
from fosco.common.consts import ActivationType


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        activation: tuple[ActivationType, ...],
        output_size: int = 1,
    ):
        super(MLP, self).__init__()
        assert len(hidden_sizes) == len(
            activation
        ), "hidden sizes and activation must have the same length"

        self.input_size = input_size
        self.output_size = output_size
        self.acts = activation
        self.layers = []

        # hidden layers
        n_prev, k = self.input_size, 1
        for n_hid in hidden_sizes:
            layer = nn.Linear(n_prev, n_hid)
            self.register_parameter(f"W{k}", layer.weight)
            self.register_parameter(f"b{k}", layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, self.output_size)
        self.register_parameter(f"W{k}", layer.weight)
        self.register_parameter(f"b{k}", layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        y = x
        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

        y = self.layers[-1](y)
        return y
