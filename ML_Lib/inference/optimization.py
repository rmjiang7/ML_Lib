import numpy as np
from ML_Lib.models.model import ProbabilityModel, DifferentiableProbabilityModel
from autograd.misc.optimizers import adam

class EvolutionaryStrategies(object):

    def __init__(self, model):
        assert(isinstance(model, ProbabilityModel))
        self.model = model
        self.params = self.model.get_params()
        self.n_params = self.params.shape[1]

    def train(self, n_pop = 50, sigma = 0.01, step_size = 0.001, num_iters = 5000, init = None, callback = None):

        # Initialize random population of parameters
        w = self.model.get_params()
        for i in range(num_iters):
            # Generate random perturbation of the population
            N = np.random.randn(n_pop, self.n_params)
            w_try = w + sigma*N
            R = self.model.log_prob(w_try)
            A = (R - np.mean(R))/np.std(R)
            approx_grad = 1/(n_pop * sigma) * np.dot(N.T, A)
            w = w + step_size*approx_grad

            if callback is not None:
                callback(w, i, approx_grad)
           
        self.model.set_params(w)
        return self.model

class MAP(object):

    def __init__(self, model):
        assert(isinstance(model, DifferentiableProbabilityModel))
        self.model = model
        self.lp = model.log_prob
        self.grad = model.grad_log_prob

    def train(self, step_size = 0.01, num_iters = 1000, verbose = False, callback = None):
        init = self.model.params
        final_params = adam(lambda x, _: -self.grad(x), init, step_size = step_size, num_iters = num_iters, callback = callback)
        self.model.set_params(final_params)
        return self.model

if __name__ == '__main__':
    from ML_Lib.models.glm import LinearRegression
    import autograd
    import autograd.numpy as agnp
    import matplotlib.pyplot as plt

    class FakeModel(ProbabilityModel):

        def __init__(self):
            self.params = np.random.randn(1,3)
            self.grad_log_prob = autograd.elementwise_grad(self.log_prob)

        def log_prob(self, params):
            return -agnp.sum(agnp.square(agnp.array([0.5,0.1,-0.3]) - params), axis = 1)

        def get_params(self):
            return self.params

        def set_params(self, params):
            self.params = params

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)

    Xt = np.vstack((np.ones(10),np.linspace(-10,10,10))).T
    yt = (5*Xt[:,1] + 2 + np.random.normal(0,1,size=10)).reshape((10,1))

    lr = LinearRegression(2)
    lr.set_data(Xt, yt)
    fm = FakeModel()

    X, Y = np.mgrid[-6:6:0.5, -6:6:0.5]
    plt.show(block=False)
    def callback(params, i, grad):
        if i % 1 == 0:
            plt.cla()
            U, V = np.zeros((X.shape[0],X.shape[0])), np.zeros((X.shape[0],X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    grade = lr.grad_log_prob(np.array([X[i,j],Y[i,j]]).reshape((1,-1)))
                    U[i,j] = grade[0,0]
                    V[i,j] = grade[0,1]
            plt.show(block=False)
            
            # Show posterior marginals.
            ax.set_xlim([-6,6])
            ax.set_ylim([-6,6])
            ax.quiver(params[0,0],params[0,1], grad[0], grad[1])
            true_grad = lr.grad_log_prob(params)
            ax.quiver(params[0,0], params[0,1], true_grad[0,0], true_grad[0,1])
            ax.quiver(X, Y, U, V, alpha = 0.5)
            ax.plot(params[:,0], params[:,1], 'b.')
            ax.plot([2],[5], 'g.')

            plt.draw()
            plt.pause(1.0/90.0)

    es = EvolutionaryStrategies(lr)
    es.train(sigma = 0.01, step_size = 0.01, callback = callback)

    #es = MAP(lr)
    #lr.set_params(np.array([[2,5]]))
    #print(lr.log_prob(lr.get_params()))
    

