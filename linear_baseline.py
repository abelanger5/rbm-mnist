from sklearn import datasets, neighbors, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from helpers.importer import import_mnist
from helpers.importer import import_test
import numpy as np

images_train, labels_train = import_mnist();

images_test, labels_test = import_test();

# based on confusion matrix tutorial: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes):
    title = "LogReg Confusion Matrix"
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


logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                           multi_class='multinomial')

logistic.C = 100; 

fit = logistic.fit(np.asarray(images_train), np.asarray(labels_train))

print('LogisticRegression score: %f' % fit.score(np.asarray(images_test), np.asarray(labels_test)));

predictions = logistic.predict(np.asarray(images_test));

plot_confusion_matrix(np.asarray(labels_test), predictions, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("RBM + logistic regression results:\n%s\n" % (metrics.classification_report(np.asarray(labels_test), predictions)))






