import numpy as np
from ML_Lib.models.model import ProbabilityModel
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

class TargetDistribution(ProbabilityModel):

    def __init__(self, n_dims, log_prob):
        self.n_dims = n_dims
        self.params = np.zeros((1,self.n_dims))
        self.full_log_prob = lambda params, x : log_prob
        self.full_grad_log_prob = lambda params, X: autograd.elementwise_grad(self.full_log_prob)
        self.log_prob = log_prob
        self.grad_log_prob = autograd.elementwise_grad(self.log_prob)

    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params
    
if __name__ == '__main__':

    import plotly.offline as pyo
    import plotly.graph_objs as go
    from ML_Lib.inference.sampling import HMC, MetropolisHastings
    from ML_Lib.inference.variational_inference import BlackBoxKLQPScore, BlackBoxKLQPReparam

    class BananaDistribution(object):

        def __init__(self, a, b, cov):
            self.a = a
            self.b = b
            self.cov = cov

        def sample(self,N = 1000):
            t = np.random.multivariate_normal(np.zeros(2), self.cov, size = (N))
            z1 = self.a*t[:,0]
            z2 = t[:,1]/self.a - self.b * (z1**2 + self.a)
            return np.array([z1,z2]).T

        def log_prob(self,z):
            t1 = z[:,0]/self.a
            t2 = (z[:,1] + self.b * (z[:,0]**2 + self.a))*self.a
            L = agnp.array([t1,t2]).T
            lpd = agsp.stats.multivariate_normal.logpdf(L, np.zeros(2), self.cov)
            return lpd

    class MixtureDistribution(object):

        def __init__(self, means):
            self.means = means
            self.dims = self.means.shape[1]
            self.n_mixes = self.means.shape[0]

        def sample(self, N = 100):
            s = []
            for i in range(N):
                z = np.random.choice(list(range(self.n_mixes)))
                s.append(np.random.multivariate_normal(self.means[z,:], np.eye(self.dims)))
            return np.array(s)

        def log_prob(self,z):
            lp = []
            for i in range(self.n_mixes):
                lp.append(agnp.exp(agsp.stats.multivariate_normal.logpdf(z, self.means[i,:], np.eye(self.dims))))
            return agnp.log(agnp.array(sum(lp)))

    mx = MixtureDistribution(np.array([[-7,-7],[10,10]]))
    #mx = BananaDistribution(4,1,np.array([[1,0.7],[0.7,1]]))
    s = mx.sample(1000)

    td = TargetDistribution(2, mx.log_prob)
    
    vi = BlackBoxKLQPReparam(td)
    vi.train(n_mc_samples = 5000) 
    vi_samples = vi.sample(1000)
    
    vi = go.Scatter(x = vi_samples[:,0], y = vi_samples[:,1], mode = 'markers', name = 'VI')
    truth = go.Scatter(x = s[:,0], y = s[:,1], mode = 'markers', name = 'Truth')
    pyo.plot([truth, vi])

    """
    m = HMC(td)
    hmc_samples = m.train(num_chains = 2, num_samples = 1000, step_size = 0.01, integration_steps = 50)

    hmc = go.Scatter(x = hmc_samples[:,0], y = hmc_samples[:,1], mode = 'markers', name = 'HMC')
    truth = go.Scatter(x = s[:,0], y = s[:,1], mode = 'markers', name = 'Truth')
    pyo.plot([truth, hmc])
    """
