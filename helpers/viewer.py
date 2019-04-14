""" 
Helps view the training and test images from MNIST
"""

from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np 
import sys

number = int(sys.argv[1]);

# import data and convert to numpy array
mndata = MNIST('../mnist');
images, labels = mndata.load_training();

images = np.asarray(images);
labels = np.asarray(labels);

# Plot the image
plt.imshow(np.reshape(images[number,:], (28, 28)), cmap='gray_r');
plt.title('Digit Label: {}'.format(labels[number]));
plt.show();