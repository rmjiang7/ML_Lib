import numpy as np
from scipy.stats import multivariate_normal
from ML_Lib.models.model import LikelihoodFreeProbabilityModel

class LikelihoodFreeInference(object):

    def __init__(self, model):
        assert(isinstance(model, LikelihoodFreeProbabilityModel))
        self.model = model

    def train(self, *args):
        raise NotImplementedException("Must implement in subclass!")

class RejectionABC(LikelihoodFreeInference):

    def train(self, accept_kernel, min_accept = None, max_samps = 1e6, worker_id = None):
        accepted_samples = []
        if min_accept is not None:
            while len(accepted_samples) < min_accept:
                params = self.model.sample_prior()
                gen_data_set = self.model.sample(params)
                if accept_kernel(gen_data_set):
                    accepted_samples.append(params)
                    if len(accepted_samples) > 0 and len(accepted_samples) % 10 == 0:
                        if worker_id is not None:
                            print("%d : %d" % (worker_id, len(accepted_samples)))
                        else:
                            print("%d" % len(accepted_samples))
        else:
            for i in range(int(max_samps)):
                params = self.model.sample_prior()
                gen_data_set = self.model.sample(params)
                if accept_kernel(gen_data_set):
                    accepted_samples.append(params)
        return np.vstack(accepted_samples)

class ABCSMC(LikelihoodFreeInference):

    def train(self, accept_eps_kernel, epsilon_schedule, samples_per_generation):
        
        T = len(epsilon_schedule)
        curr_epsilon = epsilon_schedule[0]
        accepted_samples = []
        weights = []

        # Generate initial samples
        while len(accepted_samples) < samples_per_generation:
            params = self.model.sample_prior()
            gen_data_set = self.model.sample(params)
            if accept_eps_kernel(gen_data_set, curr_epsilon):
                accepted_samples.append(params)
                weights.append(1/samples_per_generation)
        
        print(np.vstack(accepted_samples).shape)
        # For each generation, draw slightly perturbed samples and re-weight
        for i in range(1, T):
            n_steps_taken = 0
            print("Starting generation %d" % (i))

            # Determine kernel distribution based on previously accepted samples
            perturb_cov = 2 * np.diag(np.var(np.vstack(accepted_samples).T, axis = 1)) + np.diag(np.ones(accepted_samples[0].shape[0]) * 1e-4)
            new_accepted_samples = []
            new_weights = []
            curr_epsilon = epsilon_schedule[i]

            # Draw samples from previous generation, weighing each one
            while len(new_accepted_samples) < samples_per_generation:
                c = np.random.choice(samples_per_generation, p=weights)
                old_param = accepted_samples[c]
                new_param = np.random.multivariate_normal(old_param, perturb_cov)

                # If out of prior density range, reject
                if not self.model.prior_density(new_param) == 0:
                    gen_data_set = self.model.sample(new_param)
                    if accept_eps_kernel(gen_data_set, curr_epsilon):
                        new_accepted_samples.append(new_param)
                        denom = 0
                        for k in range(samples_per_generation):
                            denom += weights[k] * multivariate_normal.pdf(new_param, accepted_samples[k], perturb_cov)
                        new_weights.append(self.model.prior_density(new_param)/denom)
                    n_steps_taken += 1
            accepted_samples = new_accepted_samples
            # Renormalize weights
            weights = new_weights/sum(new_weights)
            print("Number of samples drawn: %d" % n_steps_taken)
        return np.vstack(accepted_samples)

class ParticleMarginalMetropolisHastings(LikelihoodFreeInference):

    def __init__(self):
        self.model = None
    
    def train(self, transition_kernel, transition_kernel_log_density, prior_log_density, 
              initial_distribution, initial_log_density, 
              markov_process, obs_log_likelihood, 
              M, data, N_samps):
        
        samples = []
        cur_param = (np.log(1.1),np.log(0.6))
        cur_marginal_likelihood = None
        while len(samples) < N_samps:
            new_param = transition_kernel(cur_param)

            # Sample a bootstrap transition parameterized by the transition
            # This is equivalent to sampling a markov process forward in time from 
            # a specified time and then computing the likelihood
            T = data.shape[0]
            particles = np.zeros((M,T))
            for i in range(M):
                particles[i,0] = initial_distribution(new_param)
            
            trans_densities = []
            weights = 1/M * np.ones(M)
            density = np.mean([initial_log_density(particles[j,0], new_param) for j in range(M)])
            new_marginal_likelihood = density
            for i in range(1,T):
                new_weights = np.ones(M)
                for j in range(M):
                    resampled_particles = particles[np.random.choice(list(range(M)), p = weights),i-1]
                    particles[j,i] = markov_process(resampled_particles, new_param)
                    new_weights[j] = obs_log_likelihood(data[i], particles[j,i], new_param)
                weights = np.exp(new_weights)/sum(np.exp(new_weights))
                new_marginal_likelihood += np.mean(new_weights)
            
            if cur_marginal_likelihood is not None:
                likelihood_ratio = prior_log_density(new_param) - prior_log_density(cur_param) 
                likelihood_ratio += transition_kernel_log_density(cur_param, new_param) - transition_kernel_log_density(new_param, cur_param)
                likelihood_ratio += new_marginal_likelihood - cur_marginal_likelihood
                
                if np.log(np.random.rand()) < min(likelihood_ratio, 0):
                    cur_param = new_param
                    cur_marginal_likelihood = new_marginal_likelihood
            else:
                cur_marginal_likelihood = new_marginal_likelihood
            
            samples.append(cur_param)
            
        return samples
    
class BootstrapParticleFilter(LikelihoodFreeInference):

    def __init__(self):
        self.model = None

    def train(self, initial_distribution, markov_process, obs_log_likelihood, M, data):
        T = data.shape[0]
        particles = np.zeros((M,T))
        for i in range(M):
            particles[i,0] = initial_distribution()
        
        weights = 1/M * np.ones(M)
        for i in range(1,T):
            new_weights = np.ones(M)
            for j in range(M):
                resampled_particles = particles[np.random.choice(list(range(M)), p = weights),i-1]
                particles[j,i] = markov_process(resampled_particles)
                new_weights[j] = obs_log_likelihood(data[i],particles[j,i])
            weights = np.exp(new_weights)/sum(np.exp(new_weights))
        return particles
                


        

        
