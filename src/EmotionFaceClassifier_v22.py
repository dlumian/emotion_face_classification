import glob
import ast
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from simple_cnn_v2 import create_model4, plot_confusion_matrix, to_markdown, to_markdown_with_index
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from plot_reconstruction import plot_reconstruction
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from numpy.random import seed

seed(1)
np.set_printoptions(precision=2)

class EmotionFaceClassifier():

    def __init__(self, home, cv2_path, df_path):
        self.home = home
        self.cv2_path = cv2_path
        self.df_path = df_path
        self.df_csv = ''.join([df_path, '.csv'])
        self.df_output_pkl = ''.join([df_path,'_ouput.pkl'])
        self.df_output_csv = ''.join([df_path,'_ouput.csv'])
        self.emo_dict = {0:'Angry', 1:'Disgust', 2: 'Fear', 3:'Happy', 4: 'Sad', 5:'Surprise', 6: 'Neutral'}
        self.emo_list = list(self.emo_dict.values())
        self.results_df = pd.DataFrame()
        self.n_components=10
        self.n_trees = 200
        self.seed_val = 99
        self.batch_size = 128
        self.n_epochs = 100
        self.values=[1, 5, 10, 20]
        self.flat_models = [MultinomialNB, RandomForestClassifier]
        self.flat_model_names = ['MNB', 'Random_forest']
        self.flat_models_bal = [True, False]

    
    def run_analysis(self):
        self.load_data()
        self.save_df()
        self.plot_example_images()
        self.split_x_y()
        self.balanced_split_x_y()
        self.table_of_data()        

        self.pca_analysis()
        self.pca_analysis_comparison()
        self.nmf_analysis_comparison()        
        for mdl, name in zip(self.flat_models, self.flat_model_names):
            for opt in self.flat_models_bal:
                self.run_flat_model(mdl, name, balanced=opt)

        self.bal_cnn_analysis_y_cat()
        self.bal_cnn_analysis_y_cont()
        self.cnn_analysis_y_cat()
        self.cnn_analysis_y_cont()
        self.format_results_df()

    def load_data(self):
        '''
        Will load and process data from self.df_csv if no
        processed data is found (i.e., self.df_output_csv does not exist)
        '''
        print(f'Loading data...')
        if not os.path.isfile(self.df_output_pkl):
            self.df = pd.read_csv(self.df_csv)
            print(f'Converting strings to arrays')
            self.gen_arrays()
        else:
            print(f'Processed data found...')
            print(f'Loading data from {self.df_output_pkl}')            
            self.df = pd.read_pickle(self.df_output_pkl)
            # self.df["img_array"] = self.df["img_array"].apply(lambda x: np.array(x[1:-1].split()).astype(int))

    def gen_arrays(self):
        self.df['img_array']=self.df['pixels'].apply(self.convert_pixels_to_array)

    def convert_pixels_to_array(self, pixels):
        array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
        array = np.array(array, dtype='uint8')
        return array
    
    def save_df(self):
        print(f'Saving data to {self.df_output_csv}')
        self.df.to_csv(self.df_output_csv)
        print(f'Saving data to {self.df_output_pkl}')
        self.df.to_pickle(self.df_output_pkl)

    def split_x_y(self):
        self.t1 = self.df[self.df['Usage']!='PrivateTest']
        self.t2 = self.df[self.df['Usage']=='PrivateTest']
        self.x_train = np.stack(self.t1['img_array'].values)
        self.y_train = np.stack(self.t1['emotion'].values)
        self.x_test = np.stack(self.t2['img_array'].values)
        self.y_test = np.stack(self.t2['emotion'].values)
        self.x_train_flat = self.x_train.reshape(self.x_train.shape[0],-1) 
        self.x_test_flat = self.x_test.reshape(self.x_test.shape[0], -1)
        self.x_train = np.expand_dims(self.x_train, axis=3) 
        self.x_test = np.expand_dims(self.x_test, axis=3) 
        self.n_classes = len(np.unique(self.y_train))
        self.y_train_cat = to_categorical(self.y_train, self.n_classes)
        self.y_test_cat = to_categorical(self.y_test, self.n_classes)


    def balanced_split_x_y(self):
        self.bal_df = self.t1.groupby('emotion')
        self.bal_df = self.bal_df.apply(lambda x: x.sample(self.bal_df.size().min(), random_state=self.seed_val).reset_index(drop=True))
        # self.bal_df_copy = self.bal_df.copy()
        self.bal_x_train = np.stack(self.bal_df['img_array'].values)
        self.bal_y_train = np.stack(self.bal_df['emotion'].values)
        self.bal_x_train_flat = self.bal_x_train.reshape(self.bal_x_train.shape[0],-1) 
        self.bal_x_train = np.expand_dims(self.bal_x_train, axis=3) 
        self.bal_y_train_cat = to_categorical(self.bal_y_train, self.n_classes)

    def table_of_data(self):
        x = pd.Series(self.emo_dict)
        self.data_df = x.to_frame()
        self.data_df.columns = ['Label']
        self.data_df['# train']=self.t1.groupby('emotion').size()
        self.data_df['# bal train']=self.bal_df.groupby('emotion').size()
        self.data_df['# test']=self.t2.groupby('emotion').size()
        to_markdown_with_index(self.data_df)

    def plot_example_images(self):  
        # self.df.groupby('Group_Id').apply(lambda x: x.sample(1)).reset_index(drop=True)
        fig=plt.figure(figsize=(10, 3))
        columns = 7
        rows = 1
        for i in range(1, columns*rows+1):
            emo_val = i-1
            # img = np.random.randint(10, size=(h,w))
            sel_img = self.df[self.df['emotion']==emo_val]['img_array'].sample(1)
            img = np.hstack(sel_img)
            fig.add_subplot(rows, columns, i)
            plt.gca().set_title(self.emo_list[emo_val])
            plt.imshow(img)
            plt.axis('off')
        plt.savefig('../images/example_imgs.png')
        # plt.show()
        plt.close()

    def pca_analysis(self):
        fig=plt.figure(figsize=(10, 3))
        columns = 8
        rows = 1
        self.pca = decomposition.PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(self.x_train_flat)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.pca.mean_.reshape(48, 48),
                   cmap=plt.cm.bone)
        plt.gca().set_title('Overall')
        plt.axis('off')
        for i in range(2, columns*rows+1):
            emo_val = i-2
            t1 = self.df[self.df['emotion']==emo_val]
            temp_x_train = np.stack(t1.pop('img_array').values)
            temp_x_train_flat = temp_x_train.reshape(temp_x_train.shape[0],-1)
            # print(temp_x_train_flat.shape) 
            fig.add_subplot(rows, columns, i)
            pca = decomposition.PCA(n_components=self.n_components, whiten=True)
            pca.fit(temp_x_train_flat)
            plt.imshow(pca.mean_.reshape(48, 48),
                   cmap=plt.cm.bone)
            plt.gca().set_title(self.emo_list[emo_val])
            plt.axis('off')
        plt.savefig('../images/pca_images.png')
        # plt.show()
        plt.close()


    def pca_analysis_comparison(self):
        print('Running PCA component comparisons')
        fig=plt.figure(figsize=(10, 2*len(self.values)))
        columns = 8
        rows = len(self.values)
        for indx, val in enumerate(self.values):
            # self.nmf = decomposition.NMF(n_components=val)
            # self.nmf.fit(self.x_train_flat)
            self.pca = decomposition.PCA(n_components=val, whiten=True)
            self.pca.fit(self.x_train_flat)

            fig.add_subplot(rows, columns, 1+(indx*columns))
            plt.imshow(self.pca.components_.mean(0).reshape(48, 48),
                   cmap=plt.cm.bone)
            plt.gca().set_title('Overall')
            plt.axis('off')
            for emo_val, i in enumerate(range(2, columns+1)):
                t1 = self.df[self.df['emotion']==emo_val]
                temp_x_train = np.stack(t1.pop('img_array').values)
                temp_x_train_flat = temp_x_train.reshape(temp_x_train.shape[0],-1)
                # print(temp_x_train_flat.shape) 
                fig.add_subplot(rows, columns, i+(indx*columns))
                pca = decomposition.PCA(n_components=val)
                pca.fit(temp_x_train_flat)
                plt.imshow(pca.components_.mean(0).reshape(48, 48),
                   cmap=plt.cm.bone)
                plt.gca().set_title(self.emo_list[emo_val])
                plt.axis('off')
        plt.savefig('../images/pca_images_comparison.png')
        # plt.show()
        plt.close()


    def nmf_analysis_comparison(self):
        print('Running NMF component comparisons')
        fig=plt.figure(figsize=(10, 2*len(self.values)))
        columns = 8
        rows = len(self.values)
        for indx, val in enumerate(self.values):
            self.nmf = decomposition.NMF(n_components=val)
            self.nmf.fit(self.x_train_flat)

            fig.add_subplot(rows, columns, 1+(indx*columns))
            plt.imshow(self.nmf.components_.mean(0).reshape(48, 48),
                       cmap=plt.cm.bone)
            plt.gca().set_title('Overall')
            plt.axis('off')
            for emo_val, i in enumerate(range(2, columns+1)):
                t1 = self.df[self.df['emotion']==emo_val]
                temp_x_train = np.stack(t1.pop('img_array').values)
                temp_x_train_flat = temp_x_train.reshape(temp_x_train.shape[0],-1)
                # print(temp_x_train_flat.shape) 
                fig.add_subplot(rows, columns, i+(indx*columns))
                nmf = decomposition.NMF(n_components=val)
                nmf.fit(temp_x_train_flat)
                plt.imshow(nmf.components_.mean(0).reshape(48, 48),
                       cmap=plt.cm.bone)
                plt.gca().set_title(self.emo_list[emo_val])
                plt.axis('off')
        plt.savefig('../images/nmf_images_comparison.png')
        # plt.show()
        plt.close()

    def run_flat_model(self, model, model_name, balanced=False):
        print(f'Running model: {model_name}')
        self.model = model()
        if not balanced:
            self.model.fit(self.x_train_flat, self.y_train)
            self.train_pred_y = self.model.predict(self.x_train_flat)
            self.train_pred_proba = self.model.predict_proba(self.x_train_flat)
            fname = model_name+'_not_balanced'
        else:
            self.model.fit(self.bal_x_train_flat, self.bal_y_train)
            self.train_pred_y = self.model.predict(self.bal_x_train_flat)
            self.train_pred_proba = self.model.predict_proba(self.bal_x_train_flat)
            fname = model_name+'_balanced'
        self.test_pred_y = self.model.predict(self.x_test_flat)
        self.test_pred_proba = self.model.predict_proba(self.x_test_flat)
        self.save_cm(self.test_pred_y, fname)
        self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)

    def save_cm(self, y_pred, output):
        cnf_matrix = confusion_matrix(self.y_test, y_pred.T)
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=efc.emo_list, normalize=True,
                              title=output)
        outfile = '../images/'+output+'.png'
        plt.savefig(outfile)
        # plt.show()
        plt.close()

    def cnn_analysis_y_cat(self, img_dimensions=(48,48,1)):
        self.n_classes = len(np.unique(self.y_train))
        self.model = create_model4(img_dimensions, self.n_classes)
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_datagen.fit(self.x_train)
        self.train_generator = self.train_datagen.flow(self.x_train, self.y_train_cat, 
                                                    batch_size=self.batch_size, seed=self.seed_val)
        self.test_datagen.fit(self.x_train)
        self.test_generator = self.test_datagen.flow(self.x_test, self.y_test_cat, 
                                                    batch_size=self.batch_size, seed=self.seed_val)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"] ) # (keep)
        tensorboard = TensorBoard(log_dir='./logs_cat', batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.bestmodelfilepath = "bestmodel_cat.hdf5"
        checkpoint = ModelCheckpoint(self.bestmodelfilepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(self.train_generator, validation_data=self.test_generator, 
                                validation_steps=100, epochs=self.n_epochs, 
                                steps_per_epoch=50, class_weight={0:   1.0,
                                     1:  999.00,
                                     2:   1.0,
                                     3:   0.01,
                                     4:   1.0,
                                     5:   1.0,
                                     6: 99.0},
                                callbacks = [tensorboard, earlystop, checkpoint])
        # self.model.fit(self.x_train, validation_data=self.y_train, validation_steps=100, epochs=25, steps_per_epoch=50, callbacks = [tensorboard, earlystop, checkpoint])
        self.best_model = load_model(self.bestmodelfilepath)
        self.train_pred_y = self.best_model.predict_classes(self.x_train)
        self.test_pred_y = self.best_model.predict_classes(self.x_test)
        self.train_pred_proba = self.best_model.predict_proba(self.x_train)
        self.test_pred_proba = self.best_model.predict_proba(self.x_test)
        # self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.update_results('CNN Categorical', False, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.save_cm(self.test_pred_y, 'Categorical_CNN')
        self.metrics = self.best_model.evaluate_generator(self.test_generator, steps=5)

    def cnn_analysis_y_cont(self, img_dimensions=(48,48,1)):
        self.n_classes = len(np.unique(self.y_train))
        self.model = create_model4(img_dimensions, self.n_classes)
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_datagen.fit(self.x_train)
        self.train_generator = self.train_datagen.flow(self.x_train, self.y_train, 
                                            batch_size=self.batch_size, seed=self.seed_val)
        self.test_datagen.fit(self.x_test)
        self.test_generator = self.test_datagen.flow(self.x_test, self.y_test, 
                                            batch_size=self.batch_size, seed=self.seed_val)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=["sparse_categorical_accuracy"] ) # (keep)
        tensorboard = TensorBoard(log_dir='./logs_cont', batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.bestmodelfilepathcont = "bestmodel_cont.hdf5"
        checkpoint = ModelCheckpoint(self.bestmodelfilepathcont, monitor='sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(self.train_generator, validation_data=self.test_generator, validation_steps=100, epochs=self.n_epochs, 
                                steps_per_epoch=50, class_weight={0:   1.0,
                                     1:  999.00,
                                     2:   1.0,
                                     3:   0.01,
                                     4:   1.0,
                                     5:   1.0,
                                     6: 99.0},
                                callbacks = [tensorboard, earlystop, checkpoint])
        # self.model.fit(self.x_train, validation_data=self.y_train, validation_steps=100, epochs=25, steps_per_epoch=50, callbacks = [tensorboard, earlystop, checkpoint])
        self.best_model = load_model(self.bestmodelfilepathcont)
        self.train_pred_y = self.best_model.predict_classes(self.x_train)
        
        self.test_pred_y = self.best_model.predict_classes(self.x_test)
        self.train_pred_proba = self.best_model.predict_proba(self.x_train)
        self.test_pred_proba = self.best_model.predict_proba(self.x_test)
        # self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.update_results('CNN Continuous', False, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.save_cm(self.test_pred_y, 'Continuous_CNN')

        self.metrics = self.best_model.evaluate_generator(self.test_generator, steps=5)

    def bal_cnn_analysis_y_cat(self, img_dimensions=(48,48,1)):
        self.n_classes = len(np.unique(self.y_train))
        self.model = create_model4(img_dimensions, self.n_classes)
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_datagen.fit(self.bal_x_train)
        self.train_generator = self.train_datagen.flow(self.bal_x_train, self.bal_y_train_cat, 
                                        batch_size=self.batch_size, seed=self.seed_val)
        self.test_datagen.fit(self.bal_x_train)
        self.test_generator = self.test_datagen.flow(self.x_test, self.y_test_cat, 
                                        batch_size=self.batch_size, seed=self.seed_val)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"] ) # (keep)
        tensorboard = TensorBoard(log_dir='./logs_cat', batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.bestmodelfilepath = "bal_bestmodel_cat.hdf5"
        checkpoint = ModelCheckpoint(self.bestmodelfilepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(self.train_generator, validation_data=self.test_generator, 
                                validation_steps=100, epochs=self.n_epochs, 
                                steps_per_epoch=50,
                                callbacks = [tensorboard, earlystop, checkpoint])
        # self.model.fit(self.x_train, validation_data=self.y_train, validation_steps=100, epochs=25, steps_per_epoch=50, callbacks = [tensorboard, earlystop, checkpoint])
        self.best_model = load_model(self.bestmodelfilepath)
        self.train_pred_y = self.best_model.predict_classes(self.bal_x_train)
        self.test_pred_y = self.best_model.predict_classes(self.x_test)
        # self.update_results('CNN Categorical', 'Yes', train_pred_y, self.y_pred_cat)
        self.train_pred_proba = self.best_model.predict_proba(self.bal_x_train)
        self.test_pred_proba = self.best_model.predict_proba(self.x_test)
        # self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.update_results('CNN Categorical', True, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.save_cm(self.test_pred_y, 'Categorical_CNN_bal')
        self.metrics = self.best_model.evaluate_generator(self.test_generator, steps=5)

    def bal_cnn_analysis_y_cont(self, img_dimensions=(48,48,1)):
        self.n_classes = len(np.unique(self.y_train))
        self.model = create_model4(img_dimensions, self.n_classes)
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_datagen.fit(self.bal_x_train)
        self.train_generator = self.train_datagen.flow(self.bal_x_train, self.bal_y_train, 
                                            batch_size=self.batch_size, seed=self.seed_val)
        self.test_datagen.fit(self.x_test)
        self.test_generator = self.test_datagen.flow(self.x_test, self.y_test, 
                                            batch_size=self.batch_size, seed=self.seed_val)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=["sparse_categorical_accuracy"] ) # (keep)
        tensorboard = TensorBoard(log_dir='./logs_cont', batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.bestmodelfilepathcont = "bal_bestmodel_cont.hdf5"
        checkpoint = ModelCheckpoint(self.bestmodelfilepathcont, monitor='acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(self.train_generator, validation_data=self.test_generator, validation_steps=100, epochs=self.n_epochs, 
                                steps_per_epoch=50,
                                callbacks = [tensorboard, earlystop, checkpoint])
        # self.model.fit(self.bal_x_train, validation_data=self.bal_y_train, validation_steps=100, epochs=25, steps_per_epoch=50, callbacks = [tensorboard, earlystop, checkpoint])
        self.best_model = load_model(self.bestmodelfilepathcont)
        self.train_pred_y = self.best_model.predict_classes(self.bal_x_train)
        self.test_pred_y = self.best_model.predict_classes(self.x_test)
        self.save_cm(self.test_pred_y, 'Continuous_CNN_bal')
        self.train_pred_proba = self.best_model.predict_proba(self.bal_x_train)
        self.test_pred_proba = self.best_model.predict_proba(self.x_test)
        # self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.update_results('CNN Continuous', True, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)

        # self.update_results('CNN Continuous', 'Yes', train_pred_y, self.y_pred_cont)
        self.metrics = self.best_model.evaluate_generator(self.test_generator, steps=5)

    def update_results(self, model, balanced, pred_train_y, pred_test_y, train_pred_proba, test_pred_probas):
        self.train_results = pred_train_y
        self.test_results = pred_test_y
        result_ser = pd.Series()
        result_ser['Model'] = model
        result_ser['Balanced'] = balanced
        if balanced:
            result_ser['Train Accuracy'] = accuracy_score(self.bal_y_train, pred_train_y)
            result_ser['Train Log Loss'] = log_loss(self.bal_y_train, train_pred_proba)
        else:
            result_ser['Train Accuracy'] = accuracy_score(self.y_train, pred_train_y)
            result_ser['Train Log Loss'] = log_loss(self.y_train, train_pred_proba)
        result_ser['Test Accuracy'] = accuracy_score(self.y_test, pred_test_y)
        result_ser['Test Log Loss'] = log_loss(self.y_test, test_pred_probas)
        self.results_df = self.results_df.append(result_ser, ignore_index=True)
        self.format_results_df()

    def format_results_df(self):
        self.organized_results = self.results_df[['Model', 'Balanced','Train Log Loss', 'Test Log Loss', 'Train Accuracy','Test Accuracy']]
        to_markdown(self.organized_results)

if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    cv2_path = '/home/danny/anaconda3/lib/python3.6/site-packages/cv2/data/'
    df_path = home + '../stims/fer2013/fer2013'
    if not os.path.isdir('../images'):
        os.makedirs('../images')
    efc = EmotionFaceClassifier(home, cv2_path, df_path)
    efc.run_analysis()
