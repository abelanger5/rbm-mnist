from mnist import MNIST
import numpy as np 

# import data and convert to numpy array
def import_mnist(binary=False):
	if not binary:
		print("Starting data import...");

		mndata = MNIST('./mnist');
		images, labels = mndata.load_training();

		images = np.asarray(images);
		labels = np.asarray(labels);

		print("Imported MNIST data without binarization");

		return images, labels;