import numpy as np
from scipy.stats import multivariate_normal, dirichlet
from scipy.misc import logsumexp
from ML_Lib.models.model import Model
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

class GMM(Model):

    def __init__(self, n_mixes, n_dims):
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.params = np.concatenate((1/(n_mixes) * np.ones(n_mixes), np.random.normal(0,1,size=(n_mixes, n_dims)).flatten())).reshape(1,-1)

    def unpack_params(self,params):
        self.probs = params[:, :self.n_mixes]
        self.means = params[:, self.n_mixes:].reshape((-1, self.n_mixes, self.n_dims))
        return self.probs, self.means

    def set_data(self, X):
        def log_prob(params):
            probs, means = self.unpack_params(params)
            normed_probs = agnp.exp(probs)/agnp.sum(agnp.exp(probs), axis = 1).reshape(-1,1)
            #lp = agnp.zeros(params.shape[0])
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

        self.log_prob = log_prob
        self.grad_log_prob = autograd.grad(log_prob)

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
        #probs = np.hstack((probs))
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

if __name__ == '__main__':

    import plotly.offline as pyo
    import plotly.graph_objs as go
    #from ML_Lib.inference.map import MAP
    from ML_Lib.inference.variational_inference import BlackBoxKLQPReparam

    g = GMM(2, 2)
    old_params = g.get_params()
    params = np.array([0.5,0.5,5,5,-5,-5]).reshape(1,-1)
    g.set_params(params)
    data = g.sample(100)
    g.set_params(old_params)
    
    truth = go.Scatter(x = data[:,0], y = data[:,1], mode = 'markers', name = 'truth')
    g.set_data(data)

    m = BlackBoxKLQPReparam(g)
    m.train(n_mc_samples = 20)
    X = []
    for i in range(100):
        x = g.sample_from(m.sample(1), 1)
        X.append(x[0,:])
    data_fit = np.array(X)
    fit = go.Scatter(x = data_fit[:,0], y = data_fit[:,1], mode = 'markers', name = 'fit')

    pyo.plot([truth, fit])
