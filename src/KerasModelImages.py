from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

from keras.models import load_model
from keras import backend as K
from keras import applications
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
# home = '/home/ubuntu/efc/src/'
bestmodelfilepath = home + 'CNN_cont.hdf5'

model = load_model(bestmodelfilepath)

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
# layer_idx = utils.find_layer_idx(model,'conv2d_5')

# layer_idx = utils.find_layer_idx(model,'conv2d_5')
layer_idx = utils.find_layer_idx(model,'activation_10')


# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 7
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])
plt.show()
plt.close()