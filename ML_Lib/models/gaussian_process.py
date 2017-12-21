import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
from autograd.misc.optimizers import adam
from ML_Lib.models.model import ProbabilityModel

class GaussianProcessRegression(ProbabilityModel):

    def __init__(self):
        self.params = agnp.zeros((1,4))

    def unpack_kernel_params(self, params):
        return params[0], params[2:], agnp.exp(params[1]) + 0.0001

    def cov_function(self, params, x, xp):
        output_scale = agnp.exp(params[0])
        length_scale = agnp.exp(params[1:])
        diffs = agnp.expand_dims(x / length_scale, 1) - agnp.expand_dims(xp / length_scale, 0)
        return output_scale * agnp.exp(-0.5 * agnp.sum(diffs**2, axis = 2))

    def predict(self, params, x, y, xstar):
        mean, cov_params, noise_scale = self.unpack_kernel_params(params)
        cov_f_f = self.cov_function(cov_params, xstar, xstar)
        cov_y_f = self.cov_function(cov_params, x, xstar)
        cov_y_y = self.cov_function(cov_params, x, x) + noise_scale * agnp.eye(len(y))
        pred_mean = mean + agnp.dot(agnp.linalg.solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - agnp.dot(agnp.linalg.solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(self, params, x, y):
        mean, cov_params, noise_scale = self.unpack_kernel_params(params)
        cov_y_y = self.cov_function(cov_params, x, x) + noise_scale * agnp.eye(len(y))
        prior_mean = mean * agnp.ones(len(y))
        return agsp.stats.multivariate_normal.logpdf(y, prior_mean, cov_y_y)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

import matplotlib.pyplot as plt
gp = GaussianProcessRegression()
inputs = agnp.concatenate([agnp.linspace(0,3,10),agnp.linspace(6,8,10)])
targets = (agnp.cos(inputs) + agnp.random.randn(20) * 0.1)/2
inputs = (inputs - 4.0)/2
inputs = inputs.reshape((len(inputs), 1))
init_params = 0.1 * agnp.random.randn(4)

# Set up figure.
fig = plt.figure(figsize=(12,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.show(block=False)

def callback(params, i, grad):
    print("Log likelihood {}".format(gp.log_marginal_likelihood(params,inputs, targets)))
    plt.cla()

    # Show posterior marginals.
    plot_xs = agnp.reshape(agnp.linspace(-7, 7, 300), (300,1))
    pred_mean, pred_cov = gp.predict(params, inputs, targets, plot_xs)
    marg_std = agnp.sqrt(agnp.diag(pred_cov))
    ax.plot(plot_xs, pred_mean, 'b')
    ax.fill(agnp.concatenate([plot_xs, plot_xs[::-1]]),
            agnp.concatenate([pred_mean - 1.96 * marg_std,
                           (pred_mean + 1.96 * marg_std)[::-1]]),
            alpha=.15, fc='Blue', ec='None')

    # Show samples from posterior.
    sampled_funcs = agnp.random.multivariate_normal(pred_mean, pred_cov, size=10)
    ax.plot(plot_xs, sampled_funcs.T)

    ax.plot(inputs, targets, 'kx')
    ax.set_ylim([-1.5, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    plt.pause(1.0/60.0)

print("Optimizing")
grad = autograd.elementwise_grad(gp.log_marginal_likelihood)
cov_params = adam(lambda x, i: grad(x, inputs, targets), init_params, step_size = 0.1, callback = callback)

plt.pause(10.0)
