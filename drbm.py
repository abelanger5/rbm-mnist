# external libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# homemade libraries
from helpers.importer import import_mnist
from helpers.importer import import_test
import helpers.viewer as viewer
from models.rbm import RBM

def generate_likelihood_plot(likelihoods):
    steps = np.arange(1, np.size(likelihoods) + 1)
    plt.plot(steps, likelihoods, '-r', marker='.', markerSize=8)
    plt.show()

def compute_largest_prob(image, biases_h, weights):
    maximum = 0; 
    label = 0;

    for i in range(0, 10):
        arr = 1 + np.exp(biases_h + weights[:, 784 + i] + np.dot(image, weights.T[:784, :]), dtype=np.float128)
        test = np.prod(arr, axis=0)
        if (test > maximum):
            maximum = test;
            label = i;

    return label

images, labels = import_mnist();
images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

labels = labels; 

# transform labels into one-hot encoding
labels_one_hot = [];

for i in range(0, np.size(labels, axis=0)):
    one_hot = [0] * 10;
    one_hot[labels[i]] = 1;
    labels_one_hot.append(one_hot)

train_data = np.concatenate((images, labels_one_hot), axis=1)

rbm1 = RBM(train_data, num_iter=20, num_h=450);
biases_h, biases_v, weights, likelihoods = rbm1.train();

# generate_likelihood_plot(likelihoods)

images_test, labels_test = import_test();
images_test = (images_test - np.min(images_test, 0)) / (np.max(images_test, 0) + 0.0001);
num_correct = 0;

for i in range(0, np.size(images_test, axis=0)):
    model_ans = compute_largest_prob(images_test[i], biases_h, weights)
    if (model_ans == labels_test[i]):
        num_correct += 1

print('accuracy is:')
print(num_correct / np.size(images_test, axis=0))