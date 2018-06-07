from autograd import numpy as agnp

class Distribution(object):
    
    @classmethod
    def pdf(x):
        raise NotImplementedException("Implement in subclass!")
    
    @classmethod
    def logpdf(x):
        raise NotImplementedException("Implement in subclass!")
    @classmethod
    def sample(N):
        raise NotImplementedException("Implement in subclass!")

class Normal(Distribution):
    
    @classmethod
    def pdf(self, x, mean, variance):
        return (1/agnp.sqrt(2 * agnp.pi * variance)) * agnp.exp(-(agnp.square(x - mean))/(2*variance))
    
    @classmethod
    def logpdf(self, x, mean, variance):
        return -agnp.log(agnp.sqrt(2 * agnp.pi * variance)) - (agnp.square(x - mean)/(2 * variance))
    
    @classmethod
    def sample(self, mean, variance, N = 1):
        return mean + agnp.sqrt(variance) * agnp.random.normal(0, 1, size = (N,1))

class MultivariateNormal(Distribution):

    @classmethod
    def pdf(self, x, mean, cov):
        return agnp.random.multivariate_normal.pdf(x, mean, cov)

    @classmethod
    def logpdf(self, x, mean, cov):
        return agnp.random.multivariate_normal.logpdf(x, mean, cov)

    @classmethod
    def sample(self, mean, cov, N = 1):
        return agnp.random.multivariate_normal(mean, cov, size = N)

