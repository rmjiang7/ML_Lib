import numpy as np
import matplotlib.pyplot as plt
import pystan

def check_divergences(samples):
    total_div = 0
    N = len(samples.get_sampler_params()[0]['divergent__'])
    for i in range(len(samples.get_sampler_params())):
        total_div += sum(samples.get_sampler_params()[i]['divergent__'][N//2:])
    print("Number of divergences in samples: %d" % (total_div))

def plot_pair(samples, var_1, var_2):
    all_samps = samples.extract([var_1, var_2])
    fig, ax = plt.subplots(1, 1)
    ax.scatter(all_samps[var_1], all_samps[var_2])
    ax.set_title("Scatter for %s,%s" % (var_1, var_2))
    fig.set_size_inches(8,5)
    plt.show()
    
def plot_trace(samples, var_names):
    all_samps = samples.extract(var_names)
    n_var_names = len(var_names)
    
    fig, ax = plt.subplots(n_var_names, 1, sharex = False)
    for i in range(n_var_names):
        Z = all_samps[var_names[i]]
        n_samples = Z.shape[0]
        if n_var_names == 1:
            ax.plot(list(range(n_samples)),Z)
            ax.set_title("Traceplot for %s" % var_names[i])
        else:
            ax[i].plot(list(range(n_samples)),Z)
            ax[i].set_title("Traceplot for %s" % var_names[i])
            
    fig.set_size_inches(8, 5 * n_var_names)
    plt.show()

def plot_trace_matrix(samples, var_names, idx):
    all_samps = samples.extract(var_names)
    n_var_names = len(var_names)
    
    fig, ax = plt.subplots(n_var_names, 1, sharex = False)
    for i in range(n_var_names):
        Z = all_samps[var_names[i]]
        n_samples = Z.shape[0]
        if n_var_names == 1:
            ax.plot(list(range(n_samples)),Z[:,idx[i]])
            ax.set_title("Traceplot for %s" % (var_names[i] + "_" + str(idx[i])))
        else:
            ax[i].plot(list(range(n_samples)),Z[:,idx[i]])
            ax[i].set_title("Traceplot for %s" % (var_names[i] + "_" + str(idx[i])))
    fig.set_size_inches(8, 5 * n_var_names)
    plt.show()
    
def plot_hist(samples, var_names):
    all_samps = samples.extract(var_names)
    n_var_names = len(var_names)
    
    fig, ax = plt.subplots(n_var_names, 1, sharex = False)
    for i in range(n_var_names):
        Z = all_samps[var_names[i]]
        n_samples = Z.shape[0]
        if n_var_names == 1:
            ax.hist(Z)
            ax.set_title("Histogram for %s" % (var_names[i]))
        else:
            ax[i].hist(Z)
            ax[i].set_title("Histogram for %s" % var_names[i])
    fig.set_size_inches(8, 5 * n_var_names)
    plt.show()

def plot_hist_matrix(samples, var_names, idx):
    all_samps = samples.extract(var_names)
    n_var_names = len(var_names)
    
    fig, ax = plt.subplots(n_var_names, 1, sharex = False)
    for i in range(n_var_names):
        Z = all_samps[var_names[i]]
        n_samples = Z.shape[0]
        if n_var_names == 1:
            ax.hist(Z[:,idx[i]])
            ax.set_title("Histogram for %s" % (var_names[i] + "_" + str(idx[i])))
        else:
            ax[i].hist(Z)
            ax[i].set_title("Histogram for %s" % (var_names[i] + "_" + str(idx[i])))
    fig.set_size_inches(8, 5 * n_var_names)
    plt.show()
