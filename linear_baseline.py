from sklearn import datasets, neighbors, linear_model
from helpers.importer import import_mnist
from helpers.importer import import_test
import numpy as np

images_train, labels_train = import_mnist();

images_test, labels_test = import_test();


logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=100,
                                           multi_class='multinomial')

logistic.C = 100; 

print('LogisticRegression score: %f'
      % logistic.fit(np.asarray(images_train), np.asarray(labels_train)).score(np.asarray(images_test), np.asarray(labels_test)));