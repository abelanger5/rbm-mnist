# external libraries
import time
import numpy as np 
from scipy.special import expit
from sklearn.utils import gen_even_slices

# homemade
from helpers.importer import import_mnist
import helpers.viewer as viewer
from models.rbm import RBM

images, labels = import_mnist();
images = np.floor(2 * (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001));

rbm = RBM(images);
rbm.train();

n_samples = 8; 

# loop to generate n_samples
for i in range(0, n_samples):
    # sample an image
    visible = images[i*1000]

    viewer.view_data(visible, labels[i*1000]);
    
    # perform 1000 steps of Gibbs sampling
    for j in range(0, 1000):
        hidden = rbm.sample_h_from_v(visible);
        visible = rbm.sample_v_from_h(hidden); 
        
    viewer.view_data(visible, labels[i*1000]);
