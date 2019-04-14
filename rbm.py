from mnist import MNIST
import numpy as np 

# import data and convert to numpy array
mndata = MNIST('./mnist');
images, labels = mndata.load_training();

images = np.asarray(images);
labels = np.asarray(labels);

print(images.shape, labels.shape);