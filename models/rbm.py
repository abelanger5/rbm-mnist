import time
import numpy as np 
from scipy.special import expit
from sklearn.utils import gen_even_slices

class RBM:
    def __init__(self, data, batch_size=10, num_iter=30, learning_rate=0.01, num_h=784):
        num_v = data.shape[1]

        # initialize hyperparameters
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.learning_rate = learning_rate

        # initialize hidden layer samples, components, and biases
        self.weights = np.asarray(np.random.normal(0, 0.01, (num_h, num_v)), order='F');
        self.biases_h = np.zeros(num_h, );
        self.biases_v = np.zeros(num_v, );
        self.h_samples = np.zeros((batch_size, num_h));

        self.data = data;
        self.rng = np.random; 

        return;

    # P(h=1|v)
    def mean_hidden_field(self, v):
        E = np.dot(v, self.weights.T) + self.biases_h;

        prob = expit(E, out=E);

        return prob;

    # P(v=1|h)
    def mean_visible_field(self, h):
        E = np.dot(h, self.weights) + self.biases_v;

        prob = expit(E, out=E);

        return prob;

    def sample_h_from_v(self, v):
        prob = self.mean_hidden_field(v);

        return (self.rng.random_sample(size=prob.shape) < prob);

    def sample_v_from_h(self, h):
        prob = self.mean_visible_field(h);

        # return (self.rng.random_sample(size=prob.shape) < prob);
        # Hinton suggests keeping probabilities
        return prob;

    def fit_batch(self, v_pos):
        # compute hidden and visible, positive and negative phases
        h_pos = self.mean_hidden_field(v_pos)
        v_neg = self.sample_v_from_h(self.h_samples)
        h_neg = self.mean_hidden_field(v_neg)

        epsilon = self.learning_rate / v_pos.shape[0];
         
        # update weights, hidden and visible biases using SML
        self.weights += epsilon * (np.dot(v_pos.T, h_pos).T - np.dot(h_neg.T, v_neg));
        self.biases_h += epsilon * (h_pos.sum(axis=0) - h_neg.sum(axis=0));
        self.biases_v += epsilon * (np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0));

        # create h_samples based on current markov field
        h_neg[np.random.uniform(size=h_neg.shape) < h_neg] = 1.0;
        self.h_samples = np.floor(h_neg, h_neg);

    def create_batches(self):
        n_batches = int(np.ceil(float(self.data.shape[0]) / self.batch_size));
        batches = list(gen_even_slices(n_batches * self.batch_size, n_batches, self.data.shape[0]));

        return batches; 

    def train(self):
        prev_time = time.time();

        batches = self.create_batches();

        # loop through iterations, new batch each iteration
        for i in range(1, self.num_iter + 1):
            for batch in batches:
                v_pos = self.data[batch]
                self.fit_batch(v_pos);

            print("Iteration %d, time elapsed %.2fs" % (i, time.time() - prev_time));
            prev_time = time.time();

        return self.biases_h, self.biases_v, self.weights; 
