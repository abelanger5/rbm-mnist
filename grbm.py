# external libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets, neighbors, metrics, linear_model # for rbm -> logistic regression
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# homemade libraries
from helpers.importer import import_mnist
from helpers.importer import import_test
import helpers.viewer as viewer
from models.rbm import RBM
from models.simple_generator import SimpleGenerator

def generate_sample_animation(rbm, name, images, weights, biases_v, biases_h, display=True):
    fig2 = plt.figure()

    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    x = np.arange(0, n_samples * 28)
    y = np.arange(0, 28).reshape(-1, 1)
    base = np.hypot(x, y)

    total_image = np.zeros((28, (n_samples) * 28));
    ims = []

    for i in range(0, n_samples):
        total_image[:, i*28:(i+1)*28] = np.reshape(images[i], (28, 28))

    ims.append((plt.imshow(total_image, cmap='gray'), ))

    # number of animation steps
    for i in range(0, 100):
        for j in range(0, n_samples):
            vis = np.ndarray.flatten(total_image[:, j*28:(j+1)*28]);

            hidden = rbm.sample_h_from_v(vis);
            visible = rbm.sample_v_from_h(hidden);

            total_image[:, j*28:(j+1)*28] = np.reshape(visible, (28, 28))

        ims.append((plt.imshow(total_image, cmap='gray'),))

    np.flip(ims);

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50,
                                   blit=True)
    im_ani.save(name + '_samples.gif', writer='imagemagick');

    plt.show()

def generate_noise_animation(rbm, name, weights, biases_v, biases_h, display=True):
    fig2 = plt.figure()

    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    total_image = np.zeros((28, (n_samples) * 28));
    ims = []

    for i in range(0, n_samples):
        total_image[:, i*28:(i+1)*28] = np.reshape(np.random.randint(0, 2, (784, )), (28, 28))

    ims.append((plt.imshow(total_image, cmap='gray'), ))

    # number of animation steps
    for i in range(0, 100):
        for j in range(0, n_samples):
            vis = np.ndarray.flatten(total_image[:, j*28:(j+1)*28]);

            hidden = rbm.sample_h_from_v(vis);
            visible = rbm.sample_v_from_h(hidden);

            total_image[:, j*28:(j+1)*28] = np.reshape(visible, (28, 28))

        ims.append((plt.imshow(total_image, cmap='gray'),))

    np.flip(ims);

    if (display):
        x = np.arange(0, n_samples * 28)
        y = np.arange(0, 28).reshape(-1, 1)
        base = np.hypot(x, y)

        im_ani = animation.ArtistAnimation(fig2, ims, interval=50,
                                   blit=True)

        im_ani.save(name + '_noise.gif', writer='imagemagick');

        plt.show()
    
    return ims; 

def generate_animation(rbm, name, weights, biases_v, biases_h, display=True):
    NUM_LABELS = 10; 

    fig2 = plt.figure()

    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    total_image = np.zeros(((n_images_per_sample) * 28, (n_samples) * 28));
    images, labels = import_mnist();
    images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

    generator = SimpleGenerator(weights, biases_v, biases_h, NUM_LABELS)

    generator.compute_activations(images, labels)

    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    total_image = np.zeros((28, (n_samples) * 28));
    ims = []

    for i in range(0, n_samples):
        visible = generator.get_image(i)
        visible = (visible - np.min(visible)) / (np.max(visible) + 0.0001);
        total_image[:, i*28:(i+1)*28] = np.reshape(visible, (28, 28))

    ims.append((plt.imshow(total_image, cmap='gray'), ))

    # number of animation steps
    for i in range(0, 200):
        for j in range(0, n_samples):
            vis = np.ndarray.flatten(total_image[:, j*28:(j+1)*28]);

            hidden = rbm.sample_h_from_v(vis);
            visible = rbm.sample_v_from_h(hidden);

            total_image[:, j*28:(j+1)*28] = np.reshape(visible, (28, 28))

        ims.append((plt.imshow(total_image, cmap='gray'),))

    np.flip(ims);

    if (display):
        x = np.arange(0, n_samples * 28)
        y = np.arange(0, 28).reshape(-1, 1)
        base = np.hypot(x, y)

        im_ani = animation.ArtistAnimation(fig2, ims, interval=25,
                                   blit=True)

        im_ani.save(name + '_generator.gif', writer='imagemagick');

        plt.show()
    
    return ims; 

def test_generator(rbm, weights, biases_v, biases_h):
    NUM_LABELS = 10; 
    
    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    total_image = np.zeros(((n_images_per_sample) * 28, (n_samples) * 28));
    images, labels = import_mnist();
    images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

    generator = SimpleGenerator(weights, biases_v, biases_h, NUM_LABELS)

    generator.compute_activations(images, labels)

    total_image = np.zeros(((10) * 28, (10) * 28));

    # loop to generate image samples
    for i in range(0, n_samples):
        for j in range(0, n_images_per_sample):
            # sample an image
            visible = generator.get_image(i)
            visible = (visible - np.min(visible)) / (np.max(visible) + 0.0001);

            for k in range(0, n_images_per_sample):
                for l in range(0, n_gibbs):
                    hidden = rbm.sample_h_from_v(visible);
                    visible = rbm.sample_v_from_h(hidden); 

            total_image[j*28:(j+1)*28,i*28:(i+1)*28] = np.reshape(visible, (28, 28));
            
    plt.imshow(total_image, cmap='gray');
    plt.show();

def generate_sample_plot(rbm, samples, weights, biases_v, biases_h):
    n_images_per_sample = 10; 
    n_gibbs = 50; 

    total_image = np.zeros(((n_images_per_sample) * 28, (len(samples)) * 28));

    # loop to generate random noise samples
    for i in range(0, len(samples)):
        visible = samples[i]

        for j in range(0, n_images_per_sample):
            total_image[j*28:(j+1)*28,i*28:(i+1)*28] = np.reshape(visible, (28, 28));

            for k in range(0, n_gibbs):
                hidden = rbm.sample_h_from_v(visible);
                visible = rbm.sample_v_from_h(hidden); 
            
    plt.imshow(total_image, cmap='gray');
    plt.show();

def generate_noise_plot(rbm, weights, biases_v, biases_h):
    n_samples = 10; 
    n_images_per_sample = 10; 
    n_gibbs = 50; 

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

def generate_likelihood_plot(likelihoods):
    steps = np.arange(1, np.size(likelihoods) + 1)
    plt.plot(steps, likelihoods, '-r', marker='.', markerSize=8)
    plt.xlabel('Iteration No.')
    plt.ylabel('Pseudo Log-Likelihood')
    plt.title('Pseudo Log-Likelihood for Training')
    plt.show()

# generate 40 filter plots
def generate_filter_plots(weights):
    total_image = np.zeros((4 * 28, 10 * 28))

    for i in range(0,4):
        for j in range(0,10):
            image = np.reshape(weights[10 * i + j, :], (28, 28))
            image = (image - np.min(image)) / (np.max(image) + 0.0001);
            total_image[i*28:(i+1)*28,j*28:(j+1)*28] = image

    plt.imshow(total_image, cmap='gray')
    plt.show()

# based on confusion matrix tutorial: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes):
    title = "RBM + LogReg Confusion Matrix"
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

def test_logistic_regression(weights, biases_v, biases_h):
    # create hidden representations of train and test images
    images_train, labels_train = import_mnist();
    images_train = (images_train - np.min(images_train, 0)) / (np.max(images_train, 0) + 0.0001);

    images_test, labels_test = import_test();
    images_test = (images_test - np.min(images_test, 0)) / (np.max(images_test, 0) + 0.0001);

    hidden_train = [np.ndarray.tolist(np.dot(images_train[0], weights.T) + biases_h)];

    print('constructing hidden training representations');

    for i in range(1, np.size(images_train, axis=0)):
        hidden = np.ndarray.tolist(np.dot(images_train[i], weights.T) + biases_h)

        hidden_train.append(hidden);

        if (i % 1000 == 0):
            print(i)

    hidden_test = [np.ndarray.tolist(np.dot(images_test[0], weights.T) + biases_h)];

    print('constructing hidden test representations');

    for i in range(1, np.size(images_test, axis=0)):
        hidden = np.ndarray.tolist(np.dot(images_test[i], weights.T) + biases_h)

        hidden_test.append(hidden);

    logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                               multi_class='multinomial')
    logistic.C = 100.; 

    fit = logistic.fit(np.asarray(hidden_train), np.asarray(labels_train))

    print('LogisticRegression score: %f'
      % fit.score(np.asarray(hidden_test), np.asarray(labels_test)));

    predictions = logistic.predict(np.asarray(hidden_test));

    plot_confusion_matrix(np.asarray(labels_test), predictions, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print("RBM + logistic regression results:\n%s\n" % (
        metrics.classification_report(np.asarray(labels_test), predictions)))

images, labels = import_mnist();
images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

samples = []

for i in range(0, 10):
    samples.append(images[i])

"""rbm1 = RBM(images, num_iter=1, num_h=400);
biases_h, biases_v, weights, likelihoods = rbm1.train();

generate_animation(rbm1, 'rbm1', weights, biases_v, biases_h)
generate_noise_animation(rbm1, 'rbm1', weights, biases_v, biases_h)
generate_sample_animation(rbm1, 'rbm1', samples, weights, biases_v, biases_h)

images, labels = import_mnist();
images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

rbm2 = RBM(images, num_iter=5, num_h=400);
biases_h, biases_v, weights, likelihoods = rbm2.train();

generate_animation(rbm2, 'rbm2', weights, biases_v, biases_h)
generate_noise_animation(rbm2, 'rbm2', weights, biases_v, biases_h)
generate_sample_animation(rbm2, 'rbm2', samples, weights, biases_v, biases_h)"""

rbm3 = RBM(images, num_iter=40, num_h=400);
biases_h, biases_v, weights, likelihoods = rbm3.train();

test_logistic_regression(weights, biases_v, biases_h)
test_generator(rbm3, weights, biases_v, biases_h)
generate_likelihood_plot(likelihoods)
generate_filter_plots(weights)
generate_sample_plot(rbm3, samples, weights, biases_v, biases_h)
generate_noise_plot(rbm3, weights, biases_v, biases_h)

# generate_animation(rbm3, 'rbm3', weights, biases_v, biases_h)
# generate_noise_animation(rbm3, 'rbm3', weights, biases_v, biases_h)
# generate_sample_animation(rbm3, 'rbm3', samples, weights, biases_v, biases_h)
# test_logistic_regression(weights, biases_v, biases_h)
