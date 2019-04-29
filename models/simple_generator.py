from helpers.mapping import Mapping
import numpy as np

class SimpleGenerator:
    def __init__(self, weights, biases_v, biases_h, num_labels):
        self.weights = weights;
        self.biases_v = biases_v; 
        self.biases_h = biases_h;

        self.mapping = Mapping(biases_h.size, num_labels)

        return;

    def compute_activations(self, images, labels):
        # use helper class 
        for i in range(0, np.size(images, axis=0)):
            # compute hidden based on image
            hidden = np.dot(images[i], self.weights.T) + self.biases_h
            
            # add hidden to mean activations
            self.mapping.add_hidden(hidden, labels[i])

    def get_image(self, label):
        hidden = self.mapping.get_hidden(label)

        hidden = (np.random.random_sample(size=hidden.shape) < hidden);

        image = np.dot(hidden, self.weights) + self.biases_v

        # TODO - GIBBS SAMPLING
        """ for i in range(0, 1000):
            hidden = rbm.sample_h_from_v(image);
            image = rbm.sample_v_from_h(hidden); """
        
        return image