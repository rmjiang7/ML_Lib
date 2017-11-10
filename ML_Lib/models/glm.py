import numpy as np
from ML_Lib.models.model import Model
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
from autograd.scipy.stats import norm

class GLM(Model):

    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.params = np.random.normal(0, 1, size = (1,n_dims))

    def evaluate_linear(self, params, X):
        return agnp.einsum('km,nm->kn', params, X)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class LinearRegression(GLM):

    def predict(self, params, X):
        return np.expand_dims(self.evaluate_linear(params, X),2)

    def set_data(self, X, y):
        def log_prob(params):
            y_hat = self.evaluate_linear(params, X)
            log_prior = agnp.sum(norm.logpdf(params, 0, 1), axis = 1)
            log_likelihood = agnp.sum(-(y - y_hat)**2)
            return log_likelihood + log_prior
        
        self.log_prob = log_prob
        self.grad_log_prob = autograd.grad(log_prob)

if __name__ == '__main__':

    import plotly.offline as pyo
    import plotly.graph_objs as go
    from ML_Lib.inference.map import MAP

    g = LinearRegression(2)
    X = np.vstack((np.ones(100), np.linspace(-5,5,100))).T
    y = np.dot(X,np.array([5,2]))

    g.set_data(X, y)
    print(g.get_params())
    m = MAP(g)
    m.train()
    print(g.get_params())
