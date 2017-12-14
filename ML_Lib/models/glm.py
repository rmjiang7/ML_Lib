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
        self.grad_log_prob = autograd.elementwise_grad(log_prob)
    
    def get_conjugate_posterior(self, X, y):
        mu_0 = np.zeros(self.n_dims)
        S_0 = np.eye(self.n_dims)
        S_0_inv = np.linalg.inv(S_0)
        S_n_inv = S_0_inv + X.T.dot(X)
        S_n = np.linalg.inv(S_n_inv)
        m_n = S_n.dot(S_0_inv.dot(mu_0) + X.T.dot(y)[:,0])
        #print(np.linalg.solve(X.T.dot(X),X.T.dot(y)))
        return m_n, S_n

class LogisticRegression(GLM):

    def inv_logit(self, Z):
        return 1/(1 + agnp.exp(-Z))

    def predict(self, params, X, cutoff = 0.5):
        return (np.expand_dims(self.inv_logit(self.evaluate_linear(params, X)),2) > cutoff).astype(int)

    def set_data(self, X, y):
        def log_prob(params):
            prob_hat = self.inv_logit(self.evaluate_linear(params, X))
            prob_hat_neg = 1 - prob_hat
            ll = agnp.einsum('km,m->k', agnp.log(prob_hat), y) + agnp.einsum('km,m->k', agnp.log(prob_hat_neg), 1-y)
            return ll.reshape(-1,1)
        
        self.log_prob = log_prob
        self.grad_log_prob = autograd.elementwise_grad(log_prob)
