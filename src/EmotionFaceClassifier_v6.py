import os 
import itertools
import numpy as np 
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt  
from numpy.random import seed #to set random seed
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from keras.utils import to_categorical
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
from keras import backend as K
print(K.image_data_format()) #
K.set_image_data_format('channels_last') # set format
print(K.image_data_format()) #
seed(1) #set random seed to 1
np.set_printoptions(precision=2) # set decimals for printing in plots

class EmotionFaceClassifier():
    '''
    Class for handling model building and new data classification
    '''

    def __init__(self, home, cv2_path, df_path):
        self.home = home # where script lives
        self.cv2_path = cv2_path # where face processing files can be found (from cv2)
        self.df_path = df_path # where train images live
        self.df_csv = ''.join([df_path, '.csv']) # csv file with data
        self.df_output_pkl = ''.join([df_path,'_ouput.pkl']) # pickled output file
        self.df_output_csv = ''.join([df_path,'_ouput.csv']) # csv output file
        self.emo_dict = {0:'Angry', 
                        1: 'Disgust', 
                        2: 'Fear', 
                        3: 'Happy', 
                        4: 'Sad', 
                        5: 'Surprise', 
                        6: 'Neutral'} # condiiton dict
        self.emo_list = list(self.emo_dict.values()) # labels 
        self.results_df = pd.DataFrame() # df for storing model results (empty for now)
        self.n_components=10 # components for PCA, NMF
        self.n_trees = 200 # tress for RF
        self.seed_val = 1 # Seed val for random state
        self.values=[1, 3, 5, 10] # values to examine for PCA, NMF
        self.flat_models = [MultinomialNB, RandomForestClassifier] # non CNN models to fit
        self.flat_model_names = ['MNB', 'Random_forest'] # Names of models
        self.flat_models_bal = [True, False] # True=run balanced datasets, False=run unbalanced
        # params for keras model
        self.nb_filters = 72
        self.kernel_size = (4, 4)
        self.pool_size = (2, 2)
        # self.batch_size = 128 # n of images to processed in each keras batch
        self.batch_size = 256 # n of images to processed in each keras batch
        self.n_epochs = 100 # number of keras epochs
        self.input_size = (48,48,1)
        self.validation_steps = 50
        self.steps_per_epoch = 50


    def run_analysis(self):
        self.load_data() 
        self.save_df()
        self.drop_disgust() # disgust has far fewer imgs than other emos, so dropping to improve model performance
        self.plot_example_images() # creates ../images/example_imgs.png (1 img of each emotion)
        self.split_x_y() # create train/validate/test splits on data
        self.balanced_split_x_y() # create balanced train/validate/test splits on data
        self.table_of_data() # table showing img count by emotion types (bal and unbal)         
        # self.pca_analysis() # creates ../images/pca_images.png (mean face by emo type)
        # self.pca_analysis_comparison() # uses self.values to examine PCA components by emotion type
        # self.nmf_analysis_comparison() # uses self.values to examine NMF components by emotion type       
        # ## Iterates over flat models and balance type, appends results to self.results_df
        for mdl, name in zip(self.flat_models, self.flat_model_names):
            for opt in self.flat_models_bal:
                self.run_flat_model(mdl, name, balanced=opt)
        self.run_cnn(model_name='CNN_cat', balanced=False, categorical=True)
        self.run_cnn(model_name='CNN_cat_bal', balanced=True, categorical=True)
        self.run_cnn(model_name='CNN_cont', balanced=False, categorical=False)
        self.run_cnn(model_name='CNN_cont_bal', balanced=True, categorical=False)
        self.format_results_df() # Show results in pretty format

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

    def gen_arrays(self):
        self.df['img_array']=self.df['pixels'].apply(self.convert_pixels_to_array)

    @staticmethod
    def convert_pixels_to_array(pixels):
        array = np.array([int(x) for x in pixels.split(' ')]).reshape(48,48)
        array = np.array(array, dtype='uint8')
        return array
    
    def save_df(self):
        print(f'Saving data to {self.df_output_csv}')
        self.df.to_csv(self.df_output_csv)
        print(f'Saving data to {self.df_output_pkl}')
        self.df.to_pickle(self.df_output_pkl)

    def drop_disgust(self):
        '''
        Disgust has far fewer imgs than other categories, dropping to reduce class imbalance
        '''
        self.df = self.df[self.df['emotion']!=1] # 1 = Disgust
        self.df['emotion'] = self.df.apply(self.drop_vals_over_1,axis=1) # reassigns class values accounting for drop
        self.emo_dict = {0:'Angry', 1: 'Fear', 2:'Happy', 3: 'Sad', 4:'Surprise', 5: 'Neutral'} # new dict of output labels
        self.emo_list = list(self.emo_dict.values()) # new list of class labels

    @staticmethod
    def drop_vals_over_1(row):
        if row['emotion'] > 1:
            return row['emotion'] -1
        else:
            return row['emotion']

    def split_x_y(self):
        # Split data into train, val, test
        self.train_data = self.df[self.df['Usage']=='Training']
        self.val_data = self.df[self.df['Usage']=='PublicTest']
        self.test_data = self.df[self.df['Usage']=='PrivateTest']
        # Conver data to np arrays
        self.x_train = np.stack(self.train_data['img_array'].values)
        self.y_train = np.stack(self.train_data['emotion'].values)
        self.x_val = np.stack(self.val_data['img_array'].values)
        self.y_val = np.stack(self.val_data['emotion'].values)
        self.x_test = np.stack(self.test_data['img_array'].values)
        self.y_test = np.stack(self.test_data['emotion'].values)
        # For X arrays, create flattened (2d) versions of data
        self.x_train_flat = self.x_train.reshape(self.x_train.shape[0],-1) 
        self.x_val_flat = self.x_val.reshape(self.x_val.shape[0], -1)
        self.x_test_flat = self.x_test.reshape(self.x_test.shape[0], -1)
        # For X arrays, add axis for keras input (expects 3d arrays)
        self.x_train = np.expand_dims(self.x_train, axis=3) 
        self.x_val = np.expand_dims(self.x_val, axis=3)
        self.x_test = np.expand_dims(self.x_test, axis=3) 
        # Calculate number of classes
        self.n_classes = len(np.unique(self.y_train))
        # create OHE versions of Y
        self.y_train_cat = to_categorical(self.y_train, self.n_classes)
        self.y_val_cat = to_categorical(self.y_val, self.n_classes)
        self.y_test_cat = to_categorical(self.y_test, self.n_classes)

    def balanced_split_x_y(self):
        # create balanced versions of data, equal to lowest N by category
        self.bal_df = self.train_data.groupby('emotion')
        self.bal_df = self.bal_df.apply(lambda x: x.sample(self.bal_df.size().min(), random_state=self.seed_val).reset_index(drop=True))
        # parse x, y TRAINING data (test and val data don't change)
        self.bal_x_train = np.stack(self.bal_df['img_array'].values)
        self.bal_y_train = np.stack(self.bal_df['emotion'].values)
        self.bal_x_train_flat = self.bal_x_train.reshape(self.bal_x_train.shape[0],-1) 
        self.bal_x_train = np.expand_dims(self.bal_x_train, axis=3) 
        self.bal_y_train_cat = to_categorical(self.bal_y_train, self.n_classes)

    def table_of_data(self):
        x = pd.Series(self.emo_dict)
        self.data_df = x.to_frame()
        self.data_df.columns = ['Label']
        self.data_df['# train']=self.train_data.groupby('emotion').size()
        self.data_df['# bal train']=self.bal_df.groupby('emotion').size()
        self.data_df['# validation']=self.val_data.groupby('emotion').size()
        self.data_df['# test']=self.test_data.groupby('emotion').size()
        self.to_markdown_with_index(self.data_df)

    @staticmethod
    def to_markdown(df, round_places=3):
        """Returns a markdown, rounded representation of a dataframe"""
        print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=False))

    @staticmethod
    def to_markdown_with_index(df, round_places=3):
        """Returns a markdown, rounded representation of a dataframe"""
        print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=True))

    def plot_example_images(self):  
        fig=plt.figure(figsize=(10, 3))
        columns = 6
        rows = 1
        for i in range(1, columns*rows+1):
            emo_val = i-1
            sel_img = self.df[self.df['emotion']==emo_val]['img_array'].sample(1)
            img = np.hstack(sel_img)
            fig.add_subplot(rows, columns, i)
            plt.gca().set_title(self.emo_list[emo_val])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.savefig('../images/example_imgs.png')
        # plt.show()
        plt.close()

    def pca_analysis(self):
        fig=plt.figure(figsize=(10, 3))
        columns = self.n_classes+1
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
        columns = self.n_classes+1
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
        columns = self.n_classes+1
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
        self.plot_confusion_matrix(cnf_matrix, classes=self.emo_list, normalize=True,
                              title=output)
        outfile = '../images/'+output+'.png'
        plt.savefig(outfile)
        # plt.show()
        plt.close()

    @staticmethod
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
        self.to_markdown(self.organized_results)

    def create_model(self):
        """
        Create a simple baseline CNN

        Args:
            input_size (tuple(int, int, int)): 3-dimensional size of input to model
            n_categories (int): number of classification categories

        Returns:
            keras Sequential model: model with new head
            """
        model = Sequential()
        # 2 convolutional layers followed by a pooling layer followed by dropout
        model.add(Convolution2D(self.nb_filters, 
                                self.kernel_size,
                                padding='valid',
                                input_shape=self.input_size))
        model.add(Activation('relu'))
        model.add(Convolution2D(self.nb_filters, 
                                self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))
        # # Added additional layer
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=self.pool_size))
        # model.add(Dropout(0.25))
        # transition to an mlp
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        return model

    def run_cnn(self, model_name='CNN_cat', balanced=False, categorical=True):
        self.model = self.create_model()
        if balanced:
            x_train = self.bal_x_train
        else:
            x_train = self.x_train
        if categorical and balanced:
            y_train = self.bal_y_train_cat 
            y_val = self.y_val_cat
            y_test = self.y_test_cat
            loss_fnc = 'categorical_crossentropy'
        elif categorical and ~balanced:
            y_train = self.y_train_cat 
            y_val = self.y_val_cat
            y_test = self.y_test_cat
            loss_fnc = 'categorical_crossentropy'
        elif ~categorical and balanced:
            y_train = self.bal_y_train 
            y_val = self.y_val
            y_test = self.y_test
            loss_fnc = 'sparse_categorical_crossentropy'
        else:
            y_train = self.y_train 
            y_val = self.y_val
            y_test = self.y_test
            loss_fnc = 'sparse_categorical_crossentropy'

        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #, horizontal_flip=True)
        self.val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_datagen.fit(x_train)
        self.val_datagen.fit(x_train)
        self.test_datagen.fit(x_train)


        self.train_generator = self.train_datagen.flow(x_train, 
                                                    y_train, 
                                                    batch_size=self.batch_size, 
                                                    seed=self.seed_val)

        self.val_generator = self.val_datagen.flow(self.x_val, 
                                                    y_val, 
                                                    batch_size=self.batch_size, 
                                                    seed=self.seed_val)

        self.test_generator = self.test_datagen.flow(self.x_test, 
                                                    y_test, 
                                                    batch_size=self.batch_size, 
                                                    seed=self.seed_val)
        self.model.compile(loss=loss_fnc, 
                            optimizer='rmsprop', 
                            # optimizer='adam', 
                            metrics=["accuracy"] ) # (keep)
        tb, es, cp = self.gen_callbacks(log_dir='./'+ model_name, 
                                        best_model_name= model_name +'.hdf5')

        self.model.fit_generator(self.train_generator,
                                validation_data=self.val_generator, 
                                validation_steps=self.validation_steps, 
                                epochs=self.n_epochs, 
                                steps_per_epoch=self.steps_per_epoch,
                                callbacks = [tb, es, cp])
        self.best_model = load_model(self.bestmodelfilepath)
        self.train_pred_y = self.best_model.predict_classes(x_train)
        self.test_pred_y = self.best_model.predict_classes(self.x_test)
        self.train_pred_proba = self.best_model.predict_proba(x_train)
        self.test_pred_proba = self.best_model.predict_proba(self.x_test)
        self.update_results(model_name, balanced, self.train_pred_y, self.test_pred_y, self.train_pred_proba, self.test_pred_proba)
        self.save_cm(self.test_pred_y, model_name)
        self.metrics = self.best_model.evaluate_generator(self.test_generator, steps=5)

    def gen_callbacks(self, log_dir='./logs', best_model_name='bestmodel.hdf5'):
        tensorboard = TensorBoard(log_dir=log_dir, 
                                    batch_size=self.batch_size, 
                                    write_graph=True, 
                                    write_grads=True, 
                                    write_images=True)
        earlystop = EarlyStopping(monitor='loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.bestmodelfilepath = best_model_name
        checkpoint = ModelCheckpoint(self.bestmodelfilepath, 
                                        monitor='acc', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        mode='max')
        return tensorboard, earlystop, checkpoint

if __name__=='__main__':
    # home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    home = '/home/ubuntu/efc/src/'
    cv2_path = '/home/danny/anaconda3/lib/python3.6/site-packages/cv2/data/'
    df_path = home + '../stims/fer2013/fer2013'
    if not os.path.isdir('../images'):
        os.makedirs('../images')
    efc = EmotionFaceClassifier(home, cv2_path, df_path)
    efc.run_analysis()