# external libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix

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

# based on confusion matrix tutorial: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes):
    title = "Discriminative RBM Confusion Matrix"
    cmap = plt.cm.Blues
    cm = confusion_matrix(y_true, y_pred)

    # classes = classes[unique_labels(y_true, y_pred)]
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show();
    return ax

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

rbm1 = RBM(train_data, num_iter=16, num_h=450);
biases_h, biases_v, weights, likelihoods = rbm1.train();

# generate_likelihood_plot(likelihoods)

images_test, labels_test = import_test();
images_test = (images_test - np.min(images_test, 0)) / (np.max(images_test, 0) + 0.0001);
num_correct = 0;

predictions = []

for i in range(0, np.size(images_test, axis=0)):
    model_ans = compute_largest_prob(images_test[i], biases_h, weights)
    predictions.append(model_ans)
    if (model_ans == labels_test[i]):
        num_correct += 1

plot_confusion_matrix(np.asarray(labels_test), np.asarray(predictions), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print('accuracy is:')
print(num_correct / np.size(images_test, axis=0))

