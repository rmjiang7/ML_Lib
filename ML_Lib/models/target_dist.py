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
    from ML_Lib.inference.variational_inference import BlackBoxKLQPScore, BlackBoxKLQPReparam
    from ML_Lib.inference.optimization import MAP

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

        def __init__(self, means, std = 5):
            self.means = means
            self.dims = self.means.shape[1]
            self.n_mixes = self.means.shape[0]
            self.std = 5
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)
            self.params = np.zeros((1,2))

        def sample(self, N = 100):
            s = []
            for i in range(N):
                z = np.random.choice(list(range(self.n_mixes)))
                s.append(np.random.multivariate_normal(self.means[z,:], self.std*np.eye(self.dims)))
            return np.array(s)

        def log_prob(self,z):
            lp = []
            for i in range(self.n_mixes):
                lp.append(agnp.exp(agsp.stats.multivariate_normal.logpdf(z, self.means[i,:], self.std*np.eye(self.dims))))
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

    #mx = MixtureDistribution(np.array([[-20,-20],[20,20],[15,15]]))
    #mx = BananaDistribution(1,1,np.array([[1,0.7],[0.7,1]]))
    #mx = DonutDistribution()
    #mx = SquiggleDistribution(np.array([0,0]), np.array([[2,0.25],[0.25,0.5]]))
    mx = MixtureDonutDistribution()
    """
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            Z[i,j] = mx.log_prob(np.array([x[i],y[j]]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    data = [go.Surface(x = X, y = Y, z = Z)]
    pyo.plot(data)
    """
    hmc = HMC(mx)
    s = hmc.train(num_samples = 2000, num_cores = 4, num_chains = 2, step_size = 0.05)
    #s = mx.sample(1000)
    ma = MAP(mx)
    mx_max = ma.train()
    
    td = TargetDistribution(2, mx_max.log_prob, init = mx_max.get_params())
    vi = BlackBoxKLQPScore(td)
    
    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params, i, grad):
        plt.cla()
        vi.v_dist.set_params(params)
        vi_samples = vi.sample(1000)
        
        # Show posterior marginals.
        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])

        ax.plot(s[:,0], s[:,1], 'b.')
        ax.plot(vi_samples[:,0], vi_samples[:,1], 'g.')
        ax.plot(mx_max.get_params()[:,0], mx_max.get_params()[:,1],'r.')

        plt.draw()
        plt.pause(1.0/90.0)

    vi.train(n_mc_samples = 10, step_size = 0.00001, num_iters = 1000, callback = callback) 
    vi_samples = vi.sample(1000)
    plt.pause(10.0)
