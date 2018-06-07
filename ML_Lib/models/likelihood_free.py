import numpy as np
from ML_Lib.models.model import LikelihoodFreeProbabilityModel
from ML_Lib.inference.likelihoodfree import RejectionABC, ABCSMC
from scipy.stats import norm, uniform

class LhoodFreeNormalDistribution(LikelihoodFreeProbabilityModel):

    def __init__(self, n_iid_samples):
        self.n_iid_samples = n_iid_samples

    def sample_prior(self):
        return np.array([np.random.normal(0,10), np.random.normal(0, 1)]).reshape(1,-1)

    def prior_density(self, params):
        return norm.pdf(params[:,0], 0, 10) * norm.pdf(params[:,1], 0, 1)

    def sample(self, params):
        mus = params[:,0]
        sigmas = np.exp(params[:,1])
        return np.random.normal(mus, sigmas, size = (self.n_iid_samples, 1))

class SimulationModel(object):

    def simulate(self, N):
        raise NotImplementedException("Implement in subclass!")

    def set_params(self, params):
        raise NotImplementedException("Implement in subclass!")

class GillespieModel(SimulationModel):

    def gillespie_simulate(self, initial_state, reactions, rate_fx, end_T, T_samps = None, tau_fn = None):
        step_counter = 0
        if tau_fn:
            states = [np.hstack((np.array([0]),initial_state))]
            if T_samps is not None:
                T_samps = T_samps[1:]
            cur_state = states[0]
            terminate = False
            while not terminate and cur_state[0] < end_T:
                updated_state = np.zeros(initial_state.shape[0] + 1)
                rates = rate_fx(cur_state)
                tau = tau_fn(cur_state)
                # Update time
                updated_state[0] = cur_state[0] + tau

                if not (np.array(rates) > 0).all():
                    return None

                if not (np.array(rates) < 10000).all():
                    return None
                
                # Randomly simulate the number of reactions i that occur in that interval
                num_transitions = [np.random.poisson(lam = r * tau) for r in rates]

                # Update the state based on the number of reactions of each type
                updated_state[1:] = cur_state[1:]
                for i, nt in enumerate(num_transitions):
                    updated_state[1:] += nt * reactions[i,:]
                states.append(updated_state)
                cur_state = updated_state
            return np.vstack(states)
        else: 
            states = [np.hstack((np.array([0]),initial_state))]
            if T_samps is not None:
                T_samps = T_samps[1:]
                end_T = max(T_samps)
            cur_state = states[0]
            terminate = False
            while not terminate and cur_state[0] < end_T:
                if step_counter > 200000:
                    return None
                updated_state = np.zeros(initial_state.shape[0] + 1)
                rates = rate_fx(cur_state)
                total_rate = sum(rates)
                if not np.isfinite(total_rate):
                    return None
                if total_rate == 0:
                    tau = np.inf
                else:
                    tau = np.random.exponential(1/total_rate)
                    event = np.random.choice(rates.shape[0], p = rates/total_rate)
                    step_counter += 1

                    t = reactions[event,:]
                    updated_state[1:] = cur_state[1:] + t
                updated_state[0] = cur_state[0] + tau
                if T_samps is not None:
                    while not terminate and updated_state[0] > T_samps[0]:
                        states.append(np.append(T_samps[0], cur_state[1:]))
                        T_samps = T_samps[1:]
                        if len(T_samps) == 0:
                            terminate = True
                else:
                    states.append(updated_state)
                cur_state = updated_state
            #print("Steps Taken: ", step_counter)
            return np.vstack(states)

class LotkaVolterra(GillespieModel):

    def __init__(self, initial_state, birth_rate, interaction_rate, death_rate, end_T = 30):
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.interaction_rate = interaction_rate
        self.death_rate = death_rate
        self.end_T = end_T

    def simulate(self, T_samps = None, tau_fn = None):
        reactions = np.array([[1,0],[-1,1],[0,-1]])
        def rate_fx(state):
            return np.array([self.birth_rate * state[1], 
                             self.interaction_rate * state[1] * state[2], 
                             self.death_rate * state[2]])

        return self.gillespie_simulate(self.initial_state, reactions, rate_fx, self.end_T, T_samps = T_samps, tau_fn = tau_fn)

    def set_params(self, params):
        self.birth_rate = params[0]
        self.interaction_rate = params[1]
        self.death_rate = params[2]

class BirthProcess(GillespieModel):

    def __init__(self, initial_state, birth_rate, end_T = 1000):
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.end_T = end_T

    def simulate(self, T_samps = None, tau_fn = None):
        reactions = np.array([[1]])
        def rate_fx(state):
            return np.array([self.birth_rate * state[1]])
        return self.gillespie_simulate(self.initial_state, reactions, rate_fx, self.end_T, T_samps = T_samps, tau_fn = tau_fn)

    def set_params(self, params):
        self.birth_rate = params[0]

class BirthDeathProcess(GillespieModel):
   
    def __init__(self, initial_state, birth_rate, death_rate, end_T = 1000):
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.end_T = end_T

    def simulate(self, T_samps = None, tau_fn = None):
        reactions = np.array([[1],[-1]])
        def rate_fx(state):
            return np.array([self.birth_rate * state[1], self.death_rate * state[1]])
        return self.gillespie_simulate(self.initial_state, reactions, rate_fx, self.end_T, T_samps = T_samps, tau_fn = tau_fn)

    def set_params(self, params):
        self.birth_rate = params[0]
        self.death_rate = params[1]

class SchloglSystem(GillespieModel):

    def __init__(self, initial_state, k1, k2, k3, k4, end_T = 1000):
        self.initial_state = initial_state
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.end_T = end_T

    def simulate(self, T_samps = None, tau_fn = None):
        reactions = np.array([[0,0,1],[0,0,-1],[0,0,1],[0,0,-1]])
        def rate_fx(state):
            return np.array([self.k1 * state[1] * state[3] * (state[3] - 1)/2,
                             self.k2 * state[3] * (state[3] -1) * (state[3] - 2)/6,
                             self.k3 * state[2],
                             self.k4 * state[3]])
        return self.gillespie_simulate(self.initial_state, reactions, rate_fx, self.end_T, T_samps = T_samps, tau_fn = tau_fn)

    def set_params(self, params):
        self.k1 = params[0]
        self.k2 = params[1]
        self.k3 = params[2]
        self.k4 = params[3]

class LhoodFreeWrapper(LikelihoodFreeProbabilityModel):

    def __init__(self, model, n_iid_samples):
        self.n_iid_samples = n_iid_samples
        self.model = model

    def sample_prior(self):
        return np.array([np.random.normal(0,10), np.random.normal(0, 1)]).reshape(1,-1)

    def prior_density(self, params):
        return norm.pdf(params[:,0], 0, 10) * norm.pdf(params[:,1], 0, 1)

    def sample(self, params):
        mus = params[:,0]
        sigmas = np.exp(params[:,1])
        return np.random.normal(mus, sigmas, size = (self.n_iid_samples, 1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathos.multiprocessing import ProcessingPool
    import pathos.multiprocessing as multiprocessing

    birth_rate = 1.2961140064027301
    interaction_rate = 0.002601578982568134
    death_rate = 0.3859787408650144
    LK = LotkaVolterra(np.array([50,100]),birth_rate, interaction_rate, death_rate)
    truth = LK.simulate(T_samps = np.linspace(0,15,20))
    thare_mean = np.mean(truth[:,1])
    thare_var = np.std(truth[:,1])
    tlynx_mean = np.mean(truth[:,2])
    tlynx_var = np.std(truth[:,2])

    class ModelWrapper(LikelihoodFreeProbabilityModel):

        def sample_prior(self):
            return np.array([np.random.uniform(np.log(0.5),np.log(1.5)),np.random.uniform(np.log(0.0025),np.log(0.0075)),np.random.uniform(np.log(0.2),np.log(0.7))]).reshape(1,-1)

        def prior_density(self, params):
            return uniform.pdf(params[:,0], np.log(0.5), np.log(1.5)) * uniform.pdf(params[:,1], np.log(0.0025), np.log(0.0075)) * uniform.pdf(params[:,2], np.log(0.2), np.log(0.7))

        def sample(self, params):
            LK.set_params(np.exp(params[0,:]))
            return LK.simulate(T_samps = np.linspace(0,15,20))
    
    m = ModelWrapper()
    """
    hare_pop = go.Scatter(x = truth[:,0], y = truth[:,1], name = 'Hare')
    lynx_pop = go.Scatter(x = truth[:,0], y = truth[:,2], name = 'Lynx')
    pyo.plot([hare_pop, lynx_pop])
    """

    def accept_kernel_eps(gen_data_set, eps):
        hare_pop = gen_data_set[:,1]
        lynx_pop = gen_data_set[:,2]
        hare_mean = np.mean(hare_pop)
        hare_var = np.std(hare_pop)
        lynx_mean = np.mean(lynx_pop)
        lynx_var = np.std(lynx_pop)

        sqdiff = (hare_mean - thare_mean)**2 + (hare_var - thare_var)**2 + (lynx_mean - tlynx_mean)**2 + (lynx_var - tlynx_var)**2 
        if sqdiff < eps:
            return True
        else:
            return False
    
    #lfi = RejectionABC(m)
    lfi = ABCSMC(m)

    num_cores = 4
    def worker(worker_id):
        try:
            np.random.seed()
            return lfi.train(accept_kernel_eps, [20000,10000,3000], 50)
        except Exception as e:
            raise Exception(e)
        
    with ProcessingPool(num_cores) as p:
        try: 
            res = p.map(worker,[i for i in range(num_cores)])
            p.close()
        except Exception:
            p.close()

    post = np.vstack(res)
    
    plt.hist(post[:,0], label = "Birth Rate", bins = 'auto')
    plt.hist(post[:,1], label = "Interaction Rate", bins = 'auto')
    plt.hist(post[:,2], label = "Death Rate", bins = 'auto')
    plt.axvline(x = np.log(birth_rate))
    plt.axvline(x = np.log(interaction_rate))
    plt.axvline(x = np.log(death_rate))
    plt.show()
