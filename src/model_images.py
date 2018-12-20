import cv2
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from time import sleep
from keras.models import load_model
from keras import models
from keras.utils import plot_model
from scipy import stats
from collections import Counter
from drawnow import drawnow
import pdb 

class ModelExaminer():
    '''
    Class for handling model building and new data classification
    '''
    def __init__(self, model_path):
        self.model_path = model_path
        self.emo_dict = {0:'Angry', 1: 'Fear', 2:'Happy', 3: 'Sad', 4:'Surprise', 5: 'Neutral', 99: 'No Face Detected'} # new dict of output labels
        self.emo_colors = ['red', 'grey', 'yellow', 'blue', 'orange', 'tan']
        self.x_range = list(range(6))
        self.emo_list = list(self.emo_dict.values()) # labels 

    def load_model(self):
        if os.path.exists(self.model_path):
            self.best_model = load_model(self.model_path)
        else:
            print(f'Model not found check path:\n{self.model_path}')

    def model_plot(self, output_file='CNN_model.png'):
        plot_model(self.best_model, to_file=output_file)

    def get_image(self, img_path='../stims/faces/face_201.jpg'):
        self.img = cv2.imread(img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # convert img to grayscale
        sb2 = cv2.resize(self.gray, (48, 48)) 
        sb3 = np.expand_dims(sb2, axis=3) 
        self.gray = np.array([sb3])
                

    def plot_layers(self):
        img_tensor = self.gray
        layer_outputs = [layer.output for layer in self.best_model.layers[:6]] # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=self.best_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
        activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation

        layer_names = []
        for layer in self.best_model.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot            
        images_per_row = 10
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            print(layer_name)
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            print(n_features)
            print(size)
            print(images_per_row)
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    # print(channel_image)
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    # print(channel_image)
                    std = channel_image.std()
                    print(std)
                    if std == 0:
                        channel_image /=1
                    else:
                        channel_image /= std 
                    # print(channel_image)
                    channel_image *= 64
                    channel_image += 128
                    # if layer_name == 'activation_10':
                    #     pdb.set_trace()
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()


if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    # home = '/home/ubuntu/efc/src/'
    bestmodelfilepath = home + 'CNN_cont.hdf5'
    me = ModelExaminer(bestmodelfilepath)
    me.load_model()
    me.model_plot()
    me.get_image()
    me.plot_layers()