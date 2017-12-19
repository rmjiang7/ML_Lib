import numpy as np
from scipy.stats import multivariate_normal, dirichlet
from scipy.misc import logsumexp
from ML_Lib.models.model import ProbabilityModel
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

class GMM(ProbabilityModel):

    def __init__(self, n_mixes, n_dims):
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.params = np.concatenate((1/(n_mixes) * np.ones(n_mixes), np.random.normal(0,1,size=(n_mixes, n_dims)).flatten())).reshape(1,-1)
        self.full_log_prob = autograd.elementwise_grad(self.full_log_prob)

    def unpack_params(self,params):
        self.probs = params[:, :self.n_mixes]
        self.means = params[:, self.n_mixes:].reshape((-1, self.n_mixes, self.n_dims))
        return self.probs, self.means

    def full_log_prob(self, params, X):
        probs, means = self.unpack_params(params)
        normed_probs = agnp.exp(probs)/agnp.sum(agnp.exp(probs), axis = 1).reshape(-1,1)
        lps = []
        for k in range(params.shape[0]):
            lp = []
            lprior = 0
            for i in range(self.n_mixes):
                log_likelihoods = agnp.log(normed_probs[k,i]) + agsp.stats.multivariate_normal.logpdf(X, means[k,i,:], agnp.eye(self.n_dims))
                lp.append(log_likelihoods)
                lprior += agsp.stats.multivariate_normal.logpdf(means[k,i,:], agnp.zeros(self.n_dims), agnp.eye(self.n_dims))
            lp = agnp.sum(agsp.misc.logsumexp(agnp.array(lp), axis = 0)) + lprior
            lp += agsp.stats.dirichlet.logpdf(normed_probs[k,:], agnp.ones(self.n_mixes))
            lps.append(lp)
        return agnp.array(lp)

    def set_data(self, X):
        self.log_prob = lambda params: self.full_log_prob(params, X)
        self.grad_log_prob = lambda params: self.full_grad_log_prob(params, X)

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def sample(self, N_samples):
        probs, means = self.unpack_params(self.params)
        probs = agnp.exp(probs)/agnp.sum(agnp.exp(probs), axis = 1)
        
        X = [] 
        for i in range(N_samples):
            z = np.random.choice(list(range(self.n_mixes)), p = probs[0,:])
            X.append(agnp.random.multivariate_normal(means[0,z,:], agnp.eye(self.n_dims)))
        return np.array(X)

    def sample_from(self, params, N_samples):
        probs, means = self.unpack_params(params)
        probs = agnp.exp(probs)/agnp.sum(agnp.exp(probs), axis = 1)

        X = [] 
        for i in range(N_samples):
            z = np.random.choice(list(range(self.n_mixes)), p = probs[0,:])
            X.append(agnp.random.multivariate_normal(means[0,z,:], agnp.eye(self.n_dims)))
        return np.array(X)

    def simplex_transform(self, probs):
        K = probs.shape[0]
        roll_sum = 0
        y = []
        for i in range(K-1):
            if zk != 0:
                yk = agnp.log(zk/(1. - zk)) - agnp.log(1./(K-i+1))
            else:
                yk = -agnp.inf 
            y.append(yk)
            roll_sum += probs[i]
        return np.array(y)

    def inverse_simplex_transform(self, ys):
        K = ys.shape[0] + 1
        xk = []
        roll_sum = 0
        for i in range(K-1):
            iz = ys[i] + agnp.log(1/(K - i +1))
            zk = 1./(1 + agnp.exp(-iz))
            xk.append((1 - roll_sum) * zk)
            roll_sum += xk[-1]
        return np.hstack((1 - np.sum(xk),np.array(xk)))
