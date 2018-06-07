import numpy as np
from ML_Lib.models.model import DifferentiableProbabilityModel
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

class TargetDistribution(DifferentiableProbabilityModel):

    def __init__(self, n_dims, log_prob, init = None):
        self.n_dims = n_dims
        if init is None:
            self.params = np.zeros((1,self.n_dims))
        else:
            self.params = init
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
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from ML_Lib.inference.sampling import HMC, MetropolisHastings
    from ML_Lib.inference.variational_inference import BlackBoxKLQPScore, BlackBoxKLQPReparam, FullRank, MeanField, ImplicitVI
    from ML_Lib.inference.optimization import MAP
    import os

    class BananaDistribution(DifferentiableProbabilityModel):

        def __init__(self, a, b, cov):
            self.a = a
            self.b = b
            self.cov = cov
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)
            self.params = np.ones((1,2))

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

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params

    class MixtureDistribution(DifferentiableProbabilityModel):

        def __init__(self, means, covs = None):
            self.means = means
            self.dims = self.means.shape[1]
            self.n_mixes = self.means.shape[0]
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)
            self.params = np.zeros((1,2))
            if covs is None:
                self.covs = [np.eye(self.dims)] * self.n_mixes
            else:
                self.covs = covs

        def sample(self, N = 100):
            s = []
            for i in range(N):
                z = np.random.choice(list(range(self.n_mixes)))
                s.append(np.random.multivariate_normal(self.means[z,:], self.covs[z]))
            return np.array(s)

        def log_prob(self,z):
            lp = []
            for i in range(self.n_mixes):
                lp.append(agnp.exp(agsp.stats.multivariate_normal.logpdf(z, self.means[i,:], self.covs[i])))
            return agnp.log(agnp.array(sum(lp)))

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params

    class DonutDistribution(DifferentiableProbabilityModel):

        def __init__(self, radius = 2.6, sigma2 = 0.033):
            self.radius = radius
            self.sigma2 = sigma2
            self.params = np.ones((1,2))
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)

        def log_prob(self, z):
            r = agnp.linalg.norm(z)
            return -agnp.power(r - self.radius, 2) / self.sigma2

        def get_params(self):
            return self.params
        
        def set_params(self, params):
            self.params = params

    class MixtureDonutDistribution(DifferentiableProbabilityModel):

        def __init__(self, radii = [5, 2.6], sigma2 = [0.033,0.033]):
            self.radii = radii
            self.sigma2 = sigma2
            self.params = np.ones((1,2))
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)

        def log_prob(self, z):
            r = agnp.linalg.norm(z)
            lp = []
            for i in range(len(self.radii)):
                lp.append(agnp.exp(-agnp.power(r - self.radii[i], 2) / self.sigma2[i]))
            return agnp.log(agnp.array(sum(lp)))

        def get_params(self):
            return self.params
        
        def set_params(self, params):
            self.params = params

    class SquiggleDistribution(DifferentiableProbabilityModel):

        def __init__(self, mean, cov):
            self.mean = mean
            self.cov = cov
            self.params = np.zeros((1,2))
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)

        def log_prob(self, z):
            y = agnp.array([z[:,0],z[:,1] + agnp.sin(5 * z[:,0])])
            t = agsp.stats.multivariate_normal.logpdf(y.T, self.mean, self.cov)
            return t

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params

    mx = MixtureDistribution(np.array([[-3,-3],[3,3]]),covs = [np.array([[1,0.3],[0.3,1]]), np.array([[1,0.7],[0.7,1]])])
    #mx = BananaDistribution(1,1,np.array([[1,0.7],[0.7,1]]))
    #mx = DonutDistribution()
    #mx = SquiggleDistribution(np.array([0,0]), np.array([[2,0.25],[0.25,0.5]]))
    #mx = MixtureDonutDistribution()
    
    x = np.linspace(-20,20,100)
    y = np.linspace(-20,20,100)
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            Z[i,j] = mx.log_prob(np.array([x[i],y[j]]))
    
    #hmc = HMC(mx)
    #hmc_samples = hmc.train(num_samples = 8000, num_cores = 4, num_chains = 1, step_size = 0.8, integration_steps = 40)
    #ma = MAP(mx)
    #mx_max = ma.train()
    
    #td = TargetDistribution(2, mx.log_prob, init = mx.get_params())
    #vi_mean_field = BlackBoxKLQPScore(mx,variational_distribution = MeanField)
    #vi_full_rank = BlackBoxKLQPScore(mx,variational_distribution = FullRank)
    vi_implicit = ImplicitVI(mx)
    
    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)
    
    def get_callback(vi_method):
        def callback(params, i, grad):
            if i % 100 == 0:
                plt.cla()
                #vi_method.v_dist.set_params(params)
                vi_method.generator.set_params(params)
                vi_samples = vi_method.sample(1000)
                
                # Show posterior marginals.
                ax.set_xlim([-20,20])
                ax.set_ylim([-20,20])

                #ax.plot(s[:,0], s[:,1], 'b.', label = 'target')
                ax.contourf(x, y, Z)
                vf = ax.plot(vi_samples[:,0], vi_samples[:,1], 'r.', label = 'variational samples')
                ax.legend()
                plt.draw()
                plt.pause(1.0/90.0)
        return callback

    #vi_mean_field.train(n_mc_samples = 10, step_size = 0.1, num_iters = 1000, callback = get_callback(vi_mean_field))
    #vi_full_rank.train(n_mc_samples = 10, step_size = 0.1, num_iters = 1000, callback = get_callback(vi_full_rank))
    vi_implicit.train(n_iters = 100, callback = get_callback(vi_implicit))
    plt.pause(3.0)
    
    """
    # Final Plt
    # Set up figure.
    fig = plt.figure(figsize=(20,8), facecolor='white')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    plt.cla()
    vi_mf_samples = vi_mean_field.sample(1000)
    vi_fr_samples = vi_full_rank.sample(1000)

    ax1.set_xlim([-20,20])
    ax1.set_ylim([-20,20])
    ax1.contourf(x,y,Z)
    ax2.set_xlim([-20,20])
    ax2.set_ylim([-20,20])
    ax2.contourf(x,y,Z)
    ax3.set_xlim([-20,20])
    ax3.set_ylim([-20,20])
    ax3.contourf(x,y,Z)

    vfmf = ax1.plot(vi_mf_samples[:,0], vi_mf_samples[:,1], 'r.', label = 'meanfield')
    vffr = ax2.plot(vi_fr_samples[:,0], vi_fr_samples[:,1], 'b.', label = 'fullrank')
    vfhmc = ax3.plot(hmc_samples[:,0], hmc_samples[:,1], 'g.', label = 'HMC')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.draw()
    plt.savefig('/Users/richardjiang/Documents/figs/mixture.png')
    """
