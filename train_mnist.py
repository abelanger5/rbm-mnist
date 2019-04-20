# external libraries
import time
import numpy as np 
from scipy.special import expit
from sklearn.utils import gen_even_slices
import matplotlib.pyplot as plt

# homemade
from helpers.importer import import_mnist
import helpers.viewer as viewer
from models.rbm import RBM

images, labels = import_mnist();
# images = np.floor(2 * (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001));
images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

rbm = RBM(images);
rbm.train();

n_samples = 20; 
n_images_per_sample = 10; 
n_gibbs = 10; 

total_image = np.zeros(((n_images_per_sample) * 28, (n_samples) * 28));

# loop to generate random noise samples
for i in range(0, n_samples):
    visible = np.random.randint(0, 2, (784, ));

    for j in range(0, n_images_per_sample):
        total_image[j*28:(j+1)*28,i*28:(i+1)*28] = np.reshape(visible, (28, 28));

        for k in range(0, n_gibbs):
            hidden = rbm.sample_h_from_v(visible);
            visible = rbm.sample_v_from_h(hidden); 
        
plt.imshow(total_image, cmap='gray');
plt.show();

total_image = np.zeros(((n_images_per_sample) * 28, (n_samples) * 28));

# loop to generate image samples
for i in range(0, n_samples):
    # sample an image
    visible = images[i*1000]

    for j in range(0, n_images_per_sample):
        total_image[j*28:(j+1)*28,i*28:(i+1)*28] = np.reshape(visible, (28, 28));

        for k in range(0, n_gibbs):
            hidden = rbm.sample_h_from_v(visible);
            visible = rbm.sample_v_from_h(hidden); 
        
plt.imshow(total_image, cmap='gray');
plt.show();