import numpy as np
import autograd.numpy as agnp
import autograd
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import norm
from autograd.scipy.signal import convolve
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

    def __init__(self, input_dim, output_dim, nonlinearity = lambda x: x):
        super().__init__(input_dim, output_dim)
        self.nonlinearity = nonlinearity
        self.num_weights = (self.m+1)*self.n
        
        # Xavier Initialization
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))
        
    def unpack_params(self,weights):
        num_weight_sets = len(weights)
        return weights[:, :self.m*self.n].reshape((num_weight_sets, self.m, self.n)),\
               weights[:, self.m*self.n:].reshape((num_weight_sets, 1, self.n))

    def forward(self, weights, inputs):
        if len(inputs.shape) > 3:
            inputs = inputs.reshape((inputs.shape[0],inputs.shape[1],-1))
        W, b = self.unpack_params(weights)
        return self.nonlinearity(agnp.einsum('mnd,mdo->mno', inputs, W) + b)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class ConvLayer(Layer):

    def __init__(self, input_dims, kernel_shape, num_filters, nonlinearity = lambda x: x):
        depth = input_dims[0]
        y = input_dims[1]
        x = input_dims[2]
        
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.num_filter_weights = depth * num_filters * kernel_shape[0] * kernel_shape[1]
        self.filter_weights_shape = (depth, self.num_filters, kernel_shape[0], kernel_shape[1])
        self.bias_shape = (1, num_filters, 1, 1)
            
        self.output_dim = (self.num_filters,) + self.conv_output_shape(input_dims[1:], self.kernel_shape)
         
        super().__init__(np.prod(input_dims), np.prod(self.output_dim))
        self.num_weights = self.num_filter_weights + num_filters 
        # Xavier Initialization
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

    def get_output_shape(self):
        return self.output_dim

    def unpack_params(self, weights):
        num_weight_sets = len(weights)
        return weights[:, :self.num_filter_weights].reshape((num_weight_sets,) + self.filter_weights_shape),\
               weights[:,self.num_filter_weights:].reshape((num_weight_sets,) + self.bias_shape)

    def forward(self, weights, inputs):
        # Input dims are [num_weight_sets, 
        w,b = self.unpack_params(weights)
        convs = []
        for i in range(len(w)):
            conv = convolve(inputs[i,:], w[i,:], axes=([2,3],[2,3]), dot_axes = ([1], [0]), mode = 'valid')
            conv = conv + b[i,:]
            convs.append(nonlinearity(conv))
        z = agnp.array(convs)
        return z

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
        t = len(weights)
        inputs = agnp.expand_dims(inputs, 0).repeat(t,0)
        for i, w in enumerate(self.unpack_layers(weights)):
            inputs = self.layers[i].forward(w, inputs)
        return inputs
    
    def set_data(self, X, y):

        def log_prob(weights):
            pred = self.predict(weights, X)
            log_prior = agnp.sum(norm.logpdf(weights, 0, 1), axis = 1)
            log_likelihood = agnp.sum(norm.logpdf(y, pred, 0.01), axis = 1)[:,0]
            return log_likelihood + log_prior

        self.log_prob = log_prob
        self.grad_log_prob = autograd.elementwise_grad(log_prob)

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

    
if __name__ == '__main__':
    import plotly.offline as pyo
    import plotly.graph_objs as go
    from autograd.misc.optimizers import adam
    
    """
    b = BaseNeuralNetwork()
    b.add_layer(ConvLayer([3,32,32],[5,5],2))
    b.add_layer(FCLayer(np.prod(b.layers[-1].get_output_shape()), 20, nonlinearity = lambda x: x * (x > 0)))
    b.add_layer(FCLayer(20, 10, nonlinearity = lambda x : x))

    input = np.zeros((4,3,32,32))
    output = b.predict(np.vstack((b.get_params(),b.get_params())), input)
    print(output.shape)
    #output = b.predict(b.get_params(), input)

    """
    layer_dims = [1, 20, 20, 1]
    nonlinearity = lambda x: (x > 0)*x
    
    nn = DenseNeuralNetwork(layer_dims, nonlinearity)
    input = agnp.linspace(-1,1, 100).reshape(-1,1)
    target = agnp.sin(input) + agnp.random.normal(0,1,size = input.shape)
    #nn.set_data(input, target)
    params = nn.get_params()
    #params = adam(lambda x, i: -nn.grad_log_prob(x), params, step_size = 0.1, num_iters = 3000)
    #print(nn.log_prob(params))
    output = nn.predict(np.vstack((params,params)), input)
    #scatter = go.Scatter(x = input[:,0], y = output[0,:,0], name = 'Fit')
    #truth = go.Scatter(x = input[:,0], y = target[:,0], mode = 'markers', name = 'Truth')
    #pyo.plot([scatter, truth])
