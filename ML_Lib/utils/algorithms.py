import autograd 
import autograd.numpy as agnp

def self_normalized_importance_estimate(n, fx, true_likelihood, importance_distribution, importance_likelihood):

    importance_samples = [importance_distribution() for i in range(n)]
    importance_ratios = [true_likelihood(s)/importance_likelihood(s) for s in importance_samples]
    estimate = [fx(s) * w for s, w in zip(importance_samples, importance_ratios)]
    
    return agnp.sum(estimate)/agnp.sum(importance_ratios), importance_samples, importance_ratios


