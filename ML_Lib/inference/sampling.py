import numpy as np
from autograd.optimizers import adam
from autograd.scipy.stats import multivariate_normal

class MetropolisHastings(object):

    def __init__(self, model):
        self.model = model
        self.lp = model.log_prob

    def train(self, num_samples, proposal = None, proposal_ll = None, num_warmup = None, init = None):
        
        n_params = self.model.get_params().shape[1]
        if init is None:
            init = self.model.get_params()
        if num_warmup is None:
            num_warmup = num_samples/2
        if proposal is None:
            proposal = lambda theta: np.random.multivariate_normal(theta[0,:], np.eye(n_params)).reshape(1,-1)
            proposal_ll = lambda x, theta: np.sum(multivariate_normal.logpdf(x[0,:], theta[0,:], np.eye(n_params)))

        curr = init
        curr_ll = self.model.log_prob(curr)
        chain = []
        n_accepts = 0
        n_accepts_samples = 0
        
        for i in range(num_samples):
            # Draw a new sample from the proposal
            new_sample = proposal(curr)
            new_sample_ll = self.model.log_prob(new_sample)
            
            # Metropolis correction
            new_proposal_ll = new_sample_ll + proposal_ll(curr, new_sample)
            curr_proposal_ll = curr_ll + proposal_ll(new_sample, curr)
           
            if np.log(np.random.rand()) < np.minimum(0., new_proposal_ll - curr_proposal_ll):
                curr = new_sample
                curr_ll = new_sample_ll
                n_accepts += 1
                if i > num_warmup:
                    n_accepts_samples += 1
            if i > num_warmup:
                chain.append(curr)
        print("Accept %%: %f %%" % (n_accepts/num_samples))
        print("Accept sampling %%: %f %%" % (n_accepts_samples/(num_samples - num_warmup)))
        return np.vstack(chain)

class HMC(object):

    def __init__(self, model):
        self.model = model
        self.lp = model.log_prob
        self.grad_lp = model.grad_log_prob

    def integrate_leapfrog(self, p, q, step_size = 0.01, leapfrog_steps = 20):
        p = p + step_size * self.grad_lp(q) / 2

        for i in range(leapfrog_steps):
            q = q + step_size * p
            if i != leapfrog_steps - 1:
                p = p + step_size * self.grad_lp(q)

        p = p + step_size * self.grad_lp(q) / 2
        p = -p
        return p, q

    def proposal(self, q, curr_lp, step_size = 0.01, integration_steps = 20):
        curr_q = q
        p = np.random.normal(0, 1, size = (1,q.shape[1]))
        curr_p = p

        p, q = self.integrate_leapfrog(p, q, step_size = step_size, leapfrog_steps = integration_steps)
        new_lp = -self.model.log_prob(q)

        return q, new_lp, curr_lp - new_lp + np.sum(curr_p**2)/2 - np.sum(p**2)/2

    def train(self, num_samples, step_size = 0.01, integration_steps = 20, num_warmup = None, init = None):
        
        n_params = self.model.get_params().shape[1]
        if init is None:
            init = self.model.get_params()
        if num_warmup is None:
            num_warmup = num_samples/2

        curr = init
        curr_lp = -self.model.log_prob(curr)
        chain = []
        n_accepts = 0
        n_accepts_samples = 0
        
        for i in range(num_samples):
            # Draw a new sample from the proposal
            new_sample, new_lp, mh_correction = self.proposal(curr, curr_lp, step_size = step_size, integration_steps = integration_steps)
            if np.log(np.random.rand()) < np.minimum(0., mh_correction):
                curr = new_sample
                curr_lp = new_lp
                n_accepts += 1
                if i > num_warmup:
                    n_accepts_samples += 1
            if i > num_warmup:
                chain.append(curr)
        print("Accept %%: %f %%" % (n_accepts/num_samples))
        print("Accept sampling %%: %f %%" % (n_accepts_samples/(num_samples - num_warmup)))
        return np.vstack(chain)

if __name__ == "__main__":
    from ML_Lib.models.glm import LinearRegression
   
    import plotly.offline as pyo
    import plotly.graph_objs as go
    
    g = LinearRegression(2)
    X = np.vstack((np.ones(100), np.linspace(-5,5,100))).T
    y = g.evaluate_linear(np.array([2,5]).reshape(1,-1), X)

    g.set_data(X, y)
    m = HMC(g)
    samples = m.train(num_samples = 100, step_size = 0.01, integration_steps = 20)
    """
    data = []
    for i in range(len(samples)):
        output = g.predict(samples[i], X)[0,:,0] #+ np.random.normal(0,1,size = 100)
        data.append(go.Scatter(x = X[:,1], y = output, opacity = 0.1))
    data.append(go.Scatter(x = X[:,1], y = y[:,0], mode = 'markers'))
    layout = go.Layout(showlegend = False)
    fig = go.Figure(data = data, layout = layout)
    pyo.plot(fig)
    """

