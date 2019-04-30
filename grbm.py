# external libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets, neighbors, metrics, linear_model # for rbm -> logistic regression

# homemade libraries
from helpers.importer import import_mnist
from helpers.importer import import_test
import helpers.viewer as viewer
from models.rbm import RBM
from models.simple_generator import SimpleGenerator

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

    logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=100,
                                               multi_class='multinomial')
    logistic.C = 100.; 

    logistic.fit(np.asarray(hidden_train), np.asarray(labels_train))

    predictions = logistic.predict(np.asarray(hidden_test));

    print("RBM + logistic regression results:\n%s\n" % (
        metrics.classification_report(np.asarray(labels_test), predictions)))

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

def test_generator(weights, biases_v, biases_h):
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
        # sample an image
        visible = generator.get_image(i)
        visible = (visible - np.min(visible)) / (np.max(visible) + 0.0001);

        for j in range(0, n_images_per_sample):
            total_image[j*28:(j+1)*28,i*28:(i+1)*28] = np.reshape(visible, (28, 28));

            for k in range(0, n_gibbs):
                hidden = rbm.sample_h_from_v(visible);
                visible = rbm.sample_v_from_h(hidden); 
            
    plt.imshow(total_image, cmap='gray');
    plt.show();

def generate_noise_plot(weights, biases_v, biases_h):
    n_samples = 20; 
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

images, labels = import_mnist();
images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001);

samples = []

for i in range(0, 10):
    samples.append(images[i])

rbm1 = RBM(images, num_iter=1, num_h=400);
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
generate_sample_animation(rbm2, 'rbm2', samples, weights, biases_v, biases_h)

rbm3 = RBM(images, num_iter=40, num_h=400);
biases_h, biases_v, weights, likelihoods = rbm3.train();

generate_likelihood_plot(likelihoods)
generate_filter_plots(weights)
generate_animation(rbm3, 'rbm3', weights, biases_v, biases_h)
generate_noise_animation(rbm3, 'rbm3', weights, biases_v, biases_h)
generate_sample_animation(rbm3, 'rbm3', samples, weights, biases_v, biases_h)
test_logistic_regression(weights, biases_v, biases_h)
