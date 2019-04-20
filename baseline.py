import numpy as np
from sklearn.neural_network import BernoulliRBM

from helpers.importer import import_mnist
import helpers.viewer as viewer

images, labels = import_mnist();
images = np.floor(2 * (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001));

model = BernoulliRBM(learning_rate=0.05, batch_size=10, n_iter=20, n_components=400, verbose=True);
model.fit(images);

n_samples = 5; 

# loop to generate n_samples
for i in range(0, n_samples):
	visible = images[i*1000];

	viewer.view_data(visible, labels[i*1000]);
	
	# perform 1000 steps of Gibbs sampling
	for j in range(0, 1000):
		visible = model.gibbs(visible); 

	viewer.view_data(visible, labels[i*1000]);