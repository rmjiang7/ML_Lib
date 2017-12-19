import autograd.numpy as agnp
from scipy.special import gamma, kv

class Kernel(object):
    
    def compute(self, x, y):
        raise NotImplementedException("Not Implemented!")

    def compute_params(self, params, x, y):
        raise NotImplementedException("Not Implemented!")

    def get_params(self):
        raise NotImplementedException("Not Implemented!")

    def set_params(self, params):
        raise NotImplementedException("Not Implemented!")

class Linear(Kernel):

    def __init__(self, variance, height_variance, offset):
        self.variance = variance
        self.height_variance = height_variance
        self.offset = offset
        self.params = agnp.array([self.height_variance, self.variance, self.offset]).reshape((1,-1))

    def compute(self, x, y):
        return self.compute_params(self.params, x, y)

    def compute_params(self, params, x, y):
        n_params = params.shape[0]
        var = params[:,0]
        hvar = params[:,1]
        off = params[:,2]
        return agnp.sum((hvar + var * (x - off) * (y - off)).reshape((n_params, -1)),axis = 1)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class SquaredExponential(Kernel):

    def __init__(self, length_scale, variance):
        self.length_scale = length_scale
        self.variance = variance
        self.params = agnp.array([self.length_scale, self.variance]).reshape((1,-1))

    def compute(self, x, y):
        return self.compute_params(self.params, x, y)

    def compute_params(self, params, x, y):
        n_params = params.shape[0]
        ls = params[:,0]
        var = params[:,1]
        return agnp.sum((var * agnp.exp((-(x - y)**2)/(2*(ls**2)))).reshape((n_params, -1)),axis = 1)
    
    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class Periodic(Kernel):

    def __init__(self, length_scale, variance, period):
        self.length_scale = length_scale
        self.variance = variance
        self.period = period

        self.params = agnp.array([self.length_scale, self.variance, self.period]).reshape((1,-1))

    def compute(self, x, y):
        return self.compute_params(self.params, x, y)

    def compute_params(self, params, x, y):
        n_params = params.shape[0]
        ls = params[:,0]
        var = params[:,1]
        per = params[:,2]
        return agnp.sum((var * agnp.exp(-(2 * agnp.sin(agnp.pi*agnp.abs(x - y)/per))*(ls**2))).reshape((n_params, -1)),axis = 1)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

class Matern(Kernel):

    def __init__(self, length_scale, v):
        self.length_scale = length_scale
        self.v = v

        self.params = agnp.array([self.length_scale, self.v]).reshape((1,-1))

    def compute(self, x, y):
        return self.compute_params(self.params, x, y)

    def compute_params(self, params, x, y):
        n_params = params.shape[0]
        ls = params[:,0]
        v = params[:,1]
        return ((2**(1 - v))/gamma(v))*(((agnp.sqrt(2 * v) * agnp.abs(x - y))/ls)**v)*kv(v, agnp.sqrt(2*v)*agnp.abs(x-y)/ls)

    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params

class NN(Kernel):

    def __init__(self, cov):
        self.cov = cov
       
        self.params = self.cov

    def compute(self, x, y):
        return self.compute_params(self.params, x, y)

    def compute_params(self, params, x, y):
        n_params = params.shape[0]
        vs = []
        xa = agnp.vstack((1,x))
        ya = agnp.vstack((1,y))
        for i in range(n_params):
            xyp = 2 * xa.T.dot(params[i,:,:].dot(ya))
            xx = 2 * xa.T.dot(params[i,:,:].dot(xa))
            ypyp = 2 * ya.T.dot(params[i,:,:].dot(ya))
            vs.append(agnp.sum((2/agnp.pi)*agnp.arcsin(xyp/agnp.sqrt((1 + xx) * (1 + ypyp))), axis = 1))
        return agnp.array(vs)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cov = agnp.array([[1,0],[0,1]]).reshape((1,2,2))
    p = NN(cov)
    l = agnp.linspace(-5,5,100)
    mat = agnp.zeros((100,100))
    for i in range(100):
        for j in range(100):
            mat[i,j] = p.compute(l[i],l[j])[0]

    plt.imshow(mat)
    plt.show()
