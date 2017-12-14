import numpy as np
from autograd.misc.optimizers import adam

class MAP(object):

    def __init__(self, model):
        self.model = model
        self.lp = model.log_prob
        self.grad = model.grad_log_prob

    def train(self, step_size = 0.01, num_iters = 1000, verbose = False, callback = None):
        init = self.model.params
        final_params = adam(lambda x, _: -self.grad(x), init, step_size = step_size, num_iters = num_iters, callback = callback)
        self.model.set_params(final_params)
        return self.model

if __name__ == "__main__":
    from ML_Lib.models.neural_network import NeuralNetwork
   
    import plotly.offline as pyo
    import plotly.graph_objs as go
    
    layer_dims = [1, 20, 20, 1]
    nonlinearity = lambda x: (x > 0)*x
    
    nn = NeuralNetwork(layer_dims, nonlinearity)
    input = np.linspace(-1,1, 100).reshape(-1,1)
    target = np.sin(input) + np.random.normal(0,1,size = input.shape)
    nn.set_data(input, target)
    
    print(nn.log_prob(nn.params))
    m = MAP(nn)
    m.train(num_iters = 3000)
    print(nn.log_prob(nn.params))
    
    output = nn.predict(nn.params, input)
    
    scatter = go.Scatter(x = input[:,0], y = output[0,:,0], name = 'Fit')
    truth = go.Scatter(x = input[:,0], y = target[:,0], mode = 'markers', name = 'Truth')
    pyo.plot([scatter, truth])

