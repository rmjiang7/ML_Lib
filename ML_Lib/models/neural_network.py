import numpy as np
import autograd.numpy as agnp
import autograd
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import norm
from ML_Lib.models.model import Model

class Layer(object):

    def __init__(self, input_dim, output_dim):
        self.num_weights = 0
        self.m = input_dim
        self.n = output_dim

    def forward(self, weights, inputs):
        raise NotImplementedException("Must define forward pass through the layer!")

    def get_params(self):
        raise NotImplementedException("Must define method to get params")

    def set_params(self, params):
        raise NotImplementedException("Must define method to set params")

class FCLayer(Layer):

    def __init__(self, input_dim, output_dim, nonlinearity):
        super().__init__(input_dim, output_dim)
        self.nonlinearity = nonlinearity
        self.num_weights = (self.m+1)*self.n
        
        # Xavier Initialization
        cur_idx = 0
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))
        
    def unpack_params(self,weights):
        num_weight_sets = len(weights)
        return weights[:, :self.m*self.n].reshape((num_weight_sets, self.m, self.n)),\
               weights[:, self.m*self.n:].reshape((num_weight_sets, 1, self.n))

    def forward(self, weights, inputs):
        W, b = self.unpack_params(weights)
        return self.nonlinearity(agnp.einsum('mnd,mdo->mno', inputs, W) + b)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class BaseNeuralNetwork(Model):

    def __init__(self):

        self.num_weights = 0
        self.layers = []
    
    def unpack_layers(self, weights):
        num_weight_sets = len(weights)
        for layer in self.layers:
            yield weights[:, :layer.num_weights]
            weights = weights[:, layer.num_weights:]

    def predict(self, weights, inputs):
        inputs = agnp.expand_dims(inputs, 0)
        for i, w in enumerate(self.unpack_layers(weights)):
            inputs = self.layers[i].forward(w, inputs)
        return inputs

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_weights += layer.num_weights

    def get_params(self):
        return np.hstack([layer.get_params() for layer in self.layers])

    def set_params(self, weights):
        for i, w in enumerate(self.unpack_layers(weights)):
            self.layers[i].set_params(w[0,:,:])

class DenseNeuralNetwork(BaseNeuralNetwork):

    def  __init__(self, layer_dims, nonlinearity = lambda x: (x > 0)*x):
        super().__init__()
        self.nonlinearity = nonlinearity
        
        shapes = list(zip(layer_dims[:-1], layer_dims[1:]))
        for m, n in shapes[:len(shapes)-1]:
            self.add_layer(FCLayer(m, n, self.nonlinearity))

        self.add_layer(FCLayer(shapes[-1][0], shapes[-1][1], nonlinearity = lambda x: x))

    def set_data(self, X, y):

        def log_prob(weights):
            pred = self.predict(weights, X)
            log_prior = agnp.sum(norm.logpdf(weights, 0, 1), axis = 1)
            log_likelihood = agnp.sum(norm.logpdf(y, pred, 0.01), axis = 1)[:,0]
            return log_likelihood + log_prior

        self.log_prob = log_prob
        self.grad_log_prob = autograd.elementwise_grad(log_prob)

if __name__ == '__main__':
    import plotly.offline as pyo
    import plotly.graph_objs as go
    from autograd.misc.optimizers import adam
    
    layer_dims = [1, 20, 20, 1]
    nonlinearity = lambda x: (x > 0)*x
    
    nn = DenseNeuralNetwork(layer_dims, nonlinearity)
    input = agnp.linspace(-1,1, 100).reshape(-1,1)
    target = agnp.sin(input) + agnp.random.normal(0,1,size = input.shape)
    nn.set_data(input, target)
    params = nn.get_params()
    params = adam(lambda x, i: -nn.grad_log_prob(x), params, step_size = 0.1, num_iters = 3000)
    print(nn.log_prob(params))
    output = nn.predict(params, input)
    scatter = go.Scatter(x = input[:,0], y = output[0,:,0], name = 'Fit')
    truth = go.Scatter(x = input[:,0], y = target[:,0], mode = 'markers', name = 'Truth')
    pyo.plot([scatter, truth])
    
