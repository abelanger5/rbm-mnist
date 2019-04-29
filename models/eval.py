"""
An implementation of a log-likelihood approximation to track the metrics of a 
model over time
"""
import numpy as np
from sklearn.utils.extmath import log_logistic

def free_energy(v, weights, biases_v, biases_h):
    val_1 = np.dot(v, biases_v)
    val_2 = np.dot(v, weights.T)

    fe = - val_1 - np.logaddexp(0, val_2 + biases_h).sum(axis=1)

    return fe

# inspired by sklearn pseudo-likelihood, utilizes efficient log_logistic function
def pseudo_likelihood(v, weights, biases_v, biases_h):
    corruption = (np.arange(v.shape[0]),
           np.random.randint(0, v.shape[1], v.shape[0]))

    v_copy = v.copy()
    v_copy[corruption] = 1 - v_copy[corruption]

    energy = free_energy(v, weights, biases_v, biases_h);
    energy_copy = free_energy(v_copy, weights, biases_v, biases_h);

    likelihoods = v.shape[1] * log_logistic(energy_copy - energy);

    return likelihoods.mean()