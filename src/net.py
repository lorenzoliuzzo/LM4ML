import jax
from jax import numpy as jnp

class MLPnet:
    def __init__(self, key, input_layer: int, output_layer: int, hidden_layers: list[int] = []):
        self.key = key
        self.n_in = input_layer
        self.n_out = output_layer
        self.hidden_layers = hidden_layers
        self.weights, self.biases, self.activations = self.initialize_parameters()

    def initialize_parameters(self):
        layer_sizes = [self.n_in] + self.hidden_layers + [self.n_out]
        key, *subkeys = jax.random.split(self.key, len(layer_sizes) + 1)

        # Initialize weights and biases
        weights = [jax.random.normal(subkeys[i], (layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
        biases = [jnp.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]

        # Initialize activations (default to ReLU for hidden layers, sigmoid for binary output, softmax for multiclass output)
        activations = [jax.nn.relu for _ in range(len(layer_sizes) - 2)]  
        activations.append(jax.nn.sigmoid if self.n_out == 1 else jax.nn.softmax)  # Output activation based on output layer size

        return weights, biases, activations

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = self.activations[i](x @ self.weights[i] + self.biases[i])
        return x @ self.weights[-1] + self.biases[-1]
    
# choose random seed    
seed = 42

# create the model
model = MLPnet(key=jax.random.PRNGKey(seed), input_layer=2, output_layer=2, hidden_layers=[])

# training parameters
learning_rate = 0.001
epochs = 1000

