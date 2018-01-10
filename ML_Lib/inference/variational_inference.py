import numpy as np
from scipy.stats import norm
from autograd.misc.optimizers import adam
import autograd.numpy as agnp
import autograd
from ML_Lib.models.model import ProbabilityModel, DifferentiableProbabilityModel

class VariationalDistribution(object):

    def __init__(self, model):
        self.model = model
        self.is_reparameterizable = False

    def sample(self, n_samples):
        raise NotImplementedException("Must implement in subclass!")
    
    def entropy(self, params):
        raise NotImplementedException("Must implement in subclass!")
    
    def get_params(self):
        raise NotImplementedException("Must implement in subclass!")

    def set_params(self, params):
        raise NotImplementedException("Must implement in subclass!")

    def jacobian_adjustment(self, params):
        raise NotImplementedException("Must implement in subclass!")

class MeanField(VariationalDistribution):

    def __init__(self, model, init = None):
        super().__init__(model)
        self.is_reparameterizable = True
        self.n_params = self.model.get_params().shape[1]
        if init is None:
            self.v_params = np.concatenate((np.zeros(self.n_params), -5 * np.ones(self.n_params)))
        else:
            self.v_params = np.concatenate((init[0,:], np.zeros(self.n_params)))

    def unpack_params(self, params):
        mean, log_std = params[:self.n_params], params[self.n_params:]
        return mean, log_std

    def sample(self, params, n_samples):
        mean, log_std = self.unpack_params(params)
        samples = agnp.random.randn(n_samples, self.n_params) * agnp.exp(log_std) + mean
        return samples

    def entropy(self, params):
        _, log_std = self.unpack_params(params)
        return 0.5 * self.n_params * (1.0 + agnp.log(2 * agnp.pi)) + agnp.sum(log_std)

    def fisher_diag(self, params):
        mu, log_std = self.unpack_params(params)
        return agnp.concatenate([agnp.exp(-2.*log_std), agnp.ones(len(log_std))*2])

    def jacobian_adjustment(self, params):
        _, log_std = self.unpack_params(params)
        return agnp.sum(agnp.log(agnp.abs(agnp.exp(log_std))))

    def get_params(self):
        return self.v_params
    
    def set_params(self, params):
        self.v_params = params

class FullRank(VariationalDistribution):

    def __init__(self, model, init = None):
        super().__init__(model)
        self.is_reparameterizable = True
        self.n_params = self.model.get_params().shape[1]
        if init is None:
            self.v_params = np.concatenate((np.zeros(self.n_params), (agnp.zeros((self.n_params, self.n_params)) + agnp.eye(self.n_params)).flatten()))
        else:
            self.v_params = np.concatenate((init[0,:], (agnp.zeros((self.n_params, self.n_params)) + agnp.eye(self.n_params)).flatten()))
    
    def unpack_params(self, params):
        mean, cov_sqrt = params[:self.n_params], params[self.n_params:].reshape((self.n_params, self.n_params))
        return mean, agnp.dot(cov_sqrt.T, cov_sqrt)

    def sample(self, params, n_samples):
        mean, cov = self.unpack_params(params)
        L = agnp.linalg.cholesky(cov)
        samples = mean + agnp.dot(agnp.random.randn(n_samples, self.n_params), L)
        return samples

    def entropy(self, params):
        _, cov = self.unpack_params(params)
        return 0.5 * self.n_params * (1 + agnp.log(2 * agnp.pi)) + 0.5 * agnp.log(agnp.linalg.det(cov))

    def get_params(self):
        return self.v_params

    def set_params(self, params):
        self.v_params = params

class VariationalInference(object):

    def __init__(self, model):
        self.model = model

    def sample(self, n_samples):
        raise NotImplementedException("Must implement in subclass!")
    
    def train(self, *args):
        raise NotImplementedException("Must implement in subclass!")

"""
Implements Black-Box VI using score function gradient estimates.  This DOES NOT require
differentiability of the model.  However this DOES require differentiability of the 
samples of the variational distribution w.r.t to the variatonal parameters.

This is not a reparameterization of the model but generally has a higher
variance as a result.
"""
class BlackBoxKLQPScore(VariationalInference):

    def __init__(self, model, variational_distribution = MeanField):
        assert(isinstance(model, ProbabilityModel))
        self.model = model
        self.params = model.get_params()
        self.v_dist = variational_distribution(self.model, init = self.params)
        assert(self.v_dist.is_reparameterizable)
                        
    def sample(self, n_samples):
        return self.v_dist.sample(self.v_dist.get_params(),n_samples)

    def train(self, n_mc_samples, n_elbo_samples = 20, step_size = 0.01, num_iters = 1000, callback = None):

        def variational_objective(params, var_it, n_mc_samples = n_mc_samples):
            samples = self.v_dist.sample(params, n_mc_samples)
            elbo = self.v_dist.entropy(params) + agnp.mean(self.model.log_prob(samples))
            return -elbo
        
        def cb(params, i, g):
            print("Negative ELBO: %f" % variational_objective(params, i, n_mc_samples = n_elbo_samples))
            if callback is not None:
                callback(params, i, g)
        
        grad_elbo = autograd.elementwise_grad(variational_objective)
        ret = adam(lambda x, i : grad_elbo(x,i), self.v_dist.get_params(), step_size = step_size, num_iters = num_iters, callback = cb)
        self.v_dist.set_params(ret)
        return ret

"""
Implements Black-Box VI using reparameterization gradient estimates.  This DOES require
differentiability of the model w.r.t the variational parameters as well differentiability
of the reparameterization transform of the variational parameters.  More specifically, we evaluate

\mathbf{E}_{\epsilon}[\nabla_{\nu} f(x;g_{\nu}(\epsilon))]
where \theta \sim q(\nu) = g_{\nu}(\epsilon) with a reparameterization
to simple noise.  

Since this is a reparameterization of the model, non-linear transforms require a transform
of probability measure via. the Jacobian adjustment.
"""
class BlackBoxKLQPReparam(VariationalInference):

    def __init__(self, model):
        assert(isinstance(model, DifferentiableProbabilityModel))

        self.model = model
        self.params = model.get_params()
        self.n_params = self.params.shape[1]
        self.v_params = np.concatenate((np.zeros(self.n_params), -5 * np.ones(self.n_params)))

    def sample(self, n_samples):
        mean, log_std = self.v_params[:self.n_params], self.v_params[self.n_params:]
        samples = agnp.random.randn(n_samples, self.n_params) * agnp.exp(log_std) + mean
        return samples

    def train(self, n_mc_samples, n_elbo_samples = 20, step_size = 0.01, num_iters = 1000, callback = None):
        def unpack_params(params):
            mean, log_std = params[:self.n_params], params[self.n_params:]
            return mean, log_std

        def gaussian_entropy(log_std):
            return 0.5 * self.n_params * (1.0 + agnp.log(2*agnp.pi)) + agnp.sum(log_std)

        grad_gaussian_entropy = autograd.grad(gaussian_entropy)

        def grad(params,i):
            mean, log_std = unpack_params(params)
            
            # Sample from simple distribution
            rand_samples = agnp.random.randn(n_mc_samples, self.n_params)

            # Map to variational distribution samples
            samples = rand_samples * agnp.exp(log_std) + mean

            # Chain rule to compute gradients
            glp = self.model.grad_log_prob(samples)
            reparam_grad_mean = agnp.mean(glp, axis = 0) 
            reparam_grad_sigma = agnp.mean(glp * rand_samples * agnp.exp(log_std), axis = 0) + 1
            elbo_grad = agnp.concatenate((agnp.zeros(self.n_params),grad_gaussian_entropy(log_std))) + agnp.concatenate((reparam_grad_mean, reparam_grad_sigma))
            return -elbo_grad

        ret = adam(grad, self.v_params, step_size = step_size, num_iters = num_iters, callback = callback)
        self.v_params = ret
        return ret
