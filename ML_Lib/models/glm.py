import numpy as np
from ML_Lib.models.model import DifferentiableProbabilityModel
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
from autograd.scipy.stats import norm

class GLM(DifferentiableProbabilityModel):

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

    def __init__(self,n_dims):
        super().__init__(n_dims)
        self.full_grad_log_prob = autograd.elementwise_grad(self.full_log_prob)

    def predict(self, params, X):
        return np.expand_dims(self.evaluate_linear(params, X),2)

    def full_log_prob(self, params, X, y):
        y_hat = self.evaluate_linear(params, X)
        log_prior = agnp.sum(norm.logpdf(params, 0, 1), axis = 1)
        log_likelihood = agnp.sum(-(y - y_hat)**2)
        return log_likelihood + log_prior

    def set_data(self, X, y):
        if len(y.shape) == 2:
            y = y[:,0]
        self.log_prob = lambda params: self.full_log_prob(params, X, y)
        self.grad_log_prob = lambda params: self.full_grad_log_prob(params, X, y)
    
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

    def __init__(self,n_dims):
        super().__init__(n_dims)
        self.full_grad_log_prob = autograd.elementwise_grad(self.full_log_prob)
    
    def inv_logit(self, Z):
        return 1/(1 + agnp.exp(-Z))

    def predict(self, params, X, cutoff = 0.5):
        return (np.expand_dims(self.inv_logit(self.evaluate_linear(params, X)),2) > cutoff).astype(int)

    def full_log_prob(self, params, X, y):
        prob_hat = self.inv_logit(self.evaluate_linear(params, X))
        prob_hat_neg = 1 - prob_hat
        ll = agnp.einsum('km,m->k', agnp.log(prob_hat), y) + agnp.einsum('km,m->k', agnp.log(prob_hat_neg), 1-y)
        return ll.reshape(-1,1)

    def set_data(self, X, y):
        self.log_prob = lambda params: self.full_log_prob(params, X, y)
        self.grad_log_prob = lambda params: self.full_grad_log_prob(params, X, y)

if __name__ == "__main__":

    from ML_Lib.inference.variational_inference import BlackBoxKLQPScore
    import matplotlib.pyplot as plt
    
    X_pred = np.vstack((np.ones(10), np.linspace(-10,10,10))).T
    Xt = np.vstack((np.ones(1),np.linspace(-10,10,1))).T
    yt = (5*Xt[:,1] + 2 + np.random.normal(0,1,size=1)).reshape((1,1))

    lr = LinearRegression(2)
    lr.set_data(Xt, yt)
    
    vi = BlackBoxKLQPScore(lr)
    
    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(121, frameon=False)
    ax2 = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(params, i, grad):
        plt.cla()
        vi.v_dist.set_params(params)
        vi_samples = vi.sample(1000)
        mean = np.mean(vi_samples, axis = 0).reshape((1,-1))
        y_pred = lr.predict(mean, X_pred)[0,:,0]
        
        # Show posterior marginals.
        ax.set_xlim([-7,7])
        ax.set_ylim([-7,7])

        #ax.plot(s[:,0], s[:,1], 'b.')
        ax.plot(vi_samples[:,0], vi_samples[:,1], 'g.')
        ax.plot([2],[5],'b.')
        ax2.plot(X_pred[:,1], y_pred)

        plt.draw()
        plt.pause(1.0/90.0)

    vi.train(n_mc_samples = 500, step_size = 0.1, num_iters = 1000, callback = callback) 
    vi_samples = vi.sample(1000)
    plt.pause(10.0)

