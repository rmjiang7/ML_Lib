import numpy as np
import autograd.numpy as agnp
import autograd
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import norm
from ML_Lib.models.model import Model

class NeuralNetwork(Model):
    
    def __init__(self, layer_dims, nonlinearity = lambda x: (x > 0)*x):
        self.nonlinearity = nonlinearity
        
        self.shapes = list(zip(layer_dims[:-1], layer_dims[1:]))
        self.num_weights = sum((m+1)*n for m, n in self.shapes)
        self.params = np.random.normal(0, 1, size = (1,self.num_weights))
        
        def unpack_layers(weights):
            num_weight_sets = len(weights)
            for m, n in self.shapes:
                yield weights[:, :m*n].reshape((num_weight_sets, m, n)),\
                      weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
                weights = weights[:, (m+1)*n:]

        def predictions(weights, inputs):
            inputs = agnp.expand_dims(inputs, 0)
            for W, b in unpack_layers(weights):
                outputs = agnp.einsum('mnd,mdo->mno', inputs, W) + b
                inputs = self.nonlinearity(outputs)
            return outputs

        self.predict = predictions

    def set_data(self, X, y):
        def log_prob(weights):
            pred = self.predict(weights,X)
            log_prior = agnp.sum(norm.logpdf(weights, 0, 1), axis = 1)
            log_likelihood = agnp.sum(norm.logpdf(y, pred, 0.01), axis = 1)[:,0]
            return log_likelihood + log_prior
            
        self.log_prob = log_prob
        self.grad_log_prob = autograd.grad(log_prob)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

if __name__ == '__main__':
    import plotly.offline as pyo
    import plotly.graph_objs as go
    from autograd.optimizers import adam
    
    layer_dims = [1, 20, 20, 1]
    nonlinearity = lambda x: (x > 0)*x
    
    nn = NeuralNetwork(layer_dims, nonlinearity)
    input = agnp.linspace(-1,1, 100).reshape(-1,1)
    target = agnp.sin(input) + agnp.random.normal(0,1,size = input.shape)
    nn.set_data(input, target)
    params = nn.params
    
    nn.log_prob(np.vstack((params, params)))
    """
    params = adam(lambda x, i: -nn.grad_log_prob(x), params, step_size = 0.1, num_iters = 3000)
    print(nn.log_prob(params))
    output = nn.predict(params, input)
    scatter = go.Scatter(x = input[:,0], y = output[0,:,0], name = 'Fit')
    truth = go.Scatter(x = input[:,0], y = target[:,0], mode = 'markers', name = 'Truth')
    pyo.plot([scatter, truth])
    """
    
