from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
from tabulate import tabulate

def create_model(input_size, n_categories):
    """
    Create a simple baseline CNN

    Args:
        input_size (tuple(int, int, int)): 3-dimensional size of input to model
        n_categories (int): number of classification categories

    Returns:
        keras Sequential model: model with new head
        """
    nb_filters = 48
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def to_markdown(df, round_places=3):
    """Returns a markdown, rounded representation of a dataframe"""
    print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=False))

def to_markdown_with_index(df, round_places=3):
    """Returns a markdown, rounded representation of a dataframe"""
    print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=True))

