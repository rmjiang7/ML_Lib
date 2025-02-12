import numpy as np
from autograd.scipy.stats import multivariate_normal
from pathos.multiprocessing import ProcessingPool
import pathos.multiprocessing as multiprocessing
from ML_Lib.stats.distributions import Normal
import traceback

class MCMCSampler(object):

    def __init__(self, model):
        self.model = model
        self.lp = model.log_prob

    def multichain_sampling(self, draw_samples, num_cores = None, num_chains = 1):
        if num_cores is None:
            num_cores = max(multiprocessing.cpu_count(), num_chains)
        elif num_cores > num_chains:
            num_cores = num_chains
        
        def worker(worker_id):
            try:
                return draw_samples()
            except Exception as e:
                raise Exception(e)
        
        with ProcessingPool(num_cores) as p:
            try: 
                res = p.map(worker,[i for i in range(num_chains)])
                #p.close()
            except Exception as e:
                traceback.print_stack(e)
                p.close()
        
        total_accept = 0
        total_accept_samples = 0
        total_samples = 0
        total_post_warmup = 0
        samples = []
        for v in res:
            a, acs, ns, nps, s = v
            total_accept += a
            total_accept_samples += acs
            total_samples += ns
            total_post_warmup += nps
            samples.append(s)
        
        print("Accepted Samples: %f%%" % ((total_accept_samples * 100)/(total_post_warmup)))
        return np.vstack(samples)

    def train(self, num_samples, num_chains, num_cores):
        raise NotImplementedException("Must be implemented in subclass!")

class MetropolisHastings(MCMCSampler):

    def __init__(self, model):
        super().__init__(model)

    def train(self, num_samples, num_chains = 1, num_cores = None, proposal = None, num_warmup = None, init = None):

        def draw_samples(num_samples = num_samples, proposal = proposal, num_warmup = num_warmup, init = init):
            n_params = self.model.get_params().shape[1]
            if init is None:
                init = self.model.get_params()
            if num_warmup is None:
                num_warmup = num_samples/2
            if proposal is None:
                proposal = lambda theta: np.random.multivariate_normal(theta[0,:], np.eye(n_params)).reshape(1,-1)
                proposal_ll = lambda x, theta: np.sum(np.random.multivariate_normal.logpdf(x[0,:], theta[0,:], np.eye(n_params)))

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
                    if i >= num_warmup:
                        n_accepts_samples += 1
                if i >= num_warmup:
                    chain.append(curr)
            return n_accepts, n_accepts_samples, num_samples, num_samples - num_warmup, np.vstack(chain)
        return self.multichain_sampling(draw_samples, num_chains = num_chains, num_cores = num_cores)

class HMC(MCMCSampler):

    def __init__(self, model):
        super().__init__(model)
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

    def train(self, num_samples, num_chains = 1, num_cores = None, step_size = 0.01, integration_steps = 20, num_warmup = None, init = None):
       
        def draw_samples(num_samples = num_samples, num_warmup = num_warmup, step_size = step_size, integration_steps = integration_steps, init = init):
            np.random.seed()
            n_params = self.model.get_params().shape[1]
            if init is None:
                #init = self.model.get_params()
                init = np.random.uniform(-10,10, size = self.model.get_params().shape)
                print(init)
            if num_warmup is None:
                num_warmup = num_samples/2

            curr = init
            curr_lp = -self.model.log_prob(curr)
            print(curr_lp)
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
                    if i >= num_warmup:
                        n_accepts_samples += 1
                if i >= num_warmup:
                    chain.append(curr)
            return n_accepts, n_accepts_samples, num_samples, num_samples - num_warmup, np.vstack(chain)
    
        return self.multichain_sampling(draw_samples, num_chains = num_chains, num_cores = num_cores)
