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
from scipy import stats
from collections import Counter
from drawnow import drawnow

class EmotionFacePredictor():
    '''
    Class for handling model building and new data classification
    '''
    def __init__(self, home, cv2_path, model_path):
        self.home = home # where script lives
        self.cv2_path = cv2_path # where face processing files can be found (from cv2)
        self.cascade_file = self.cv2_path+'haarcascade_frontalface_alt.xml'
        self.model_path = model_path
        self.emo_dict = {0:'Angry', 1: 'Fear', 2:'Happy', 3: 'Sad', 4:'Surprise', 5: 'Neutral', 99: 'No Face Detected'} # new dict of output labels
        self.x_range = list(range(6))
        self.emo_list = list(self.emo_dict.values()) # labels 

    def run_setup(self):
        self.load_model()
        if not os.path.isdir('../images'):
            os.makedirs('../images')
        self.load_face_cascade()
        self.best_model._make_predict_function()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.best_model = load_model(self.model_path)
        else:
            print(f'Model not found check path:\n{self.model_path}')

    def load_face_cascade(self):
        if os.path.exists(self.cascade_file):
            self.faceCascade = cv2.CascadeClassifier(self.cascade_file)
        else:
            print(f'Model not found check path:\n{self.cascade_file}')

    def classify_faces_image(self, img):
        self.img = cv2.imread(img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # convert img to grayscale
        faces = self.faceCascade.detectMultiScale(
            self.gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(f'Found {len(faces)} faces')
        if len(faces)>0:
            # Create array to average responses
            face_paths = []
            df_probas = []
            df_predict = []
            cnt = 1
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(self.gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.sub_face = self.gray[y:y+h, x:x+w]
                sb2 = cv2.resize(self.sub_face, (48, 48)) 
                sb3 = np.expand_dims(sb2, axis=3) 
                sb4 = np.array([sb3])
                f_path = './static/images/face_'+str(cnt)+'.png'
                cv2.imwrite(f_path, self.sub_face)
                face_paths.append(f_path)
                self.test_pred_y = self.best_model.predict_classes(sb4)
                self.test_pred_proba = self.best_model.predict_proba(sb4)
                print(self.test_pred_y)
                print(self.test_pred_proba)
                print(self.emo_dict[self.test_pred_y[0]])
                cnt +=1 
                df_probas.append(self.test_pred_proba)
                df_predict.append(self.test_pred_y)
            print('I SHOULD BE RETURNING STUFF')
            return (face_paths, np.array(df_predict), np.array(df_probas))
        else:
            print('No faces found!')
            return None

    def classify_faces_video(self, duration=15, write_imgs=False, output_plot='../images/vid_plot.png'):
        self.capture_duration = duration
        start_time = time.time()
        video_capture = cv2.VideoCapture(0)
        self.total_df_probas = []
        self.total_df_predict = []
        plt.ion()
        # self.results_df
        while( int(time.time() - start_time) < self.capture_duration ):
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f'Found {len(faces)} faces')
            plt.clf()
            if len(faces)>0:
                self.temp_df_probas = []
                self.temp_df_predict = []
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    sub_face = frame[y:y+h, x:x+w]
                    if write_imgs:
                        face_file_name = "faces/face_" + str(y) + ".jpg"
                        cv2.imwrite(face_file_name, sub_face)
                    gray_image = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                    sb2 = cv2.resize(gray_image, (48, 48)) 
                    sb3 = np.expand_dims(sb2, axis=3) 
                    sb4 = np.array([sb3])

                    self.test_pred_y = self.best_model.predict_classes(sb4)
                    self.test_pred_proba = self.best_model.predict_proba(sb4)
                    print(self.test_pred_y)
                    print(self.test_pred_proba)
                    print(self.emo_dict[self.test_pred_y[0]])

                    self.temp_df_probas.append(self.test_pred_proba)
                    self.temp_df_predict.append(self.test_pred_y[0])
                self.total_df_probas.append(np.array(self.temp_df_probas).mean(axis=0)[0])
                mode = Counter(self.temp_df_predict).most_common(1)
                self.total_df_predict.append(mode[0][0])
            else:
                self.total_df_probas.append(np.array([0, 0, 0, 0, 0, 0]))
                self.total_df_predict.append(np.NaN)
                self.test_pred_y =[99]
            self.means_to_plot = np.array(self.total_df_probas).mean(0)
            plt.bar(self.x_range, self.means_to_plot)
            # sleep(.5)
            # plt.pause(0.5)
            plt.title(self.emo_dict[self.test_pred_y[0]])
            plt.xticks(range(6), list(self.emo_dict.values()))
            plt.ylim(0,1)
            cv2.imshow('Video', frame)
            plt.draw()
            plt.pause(.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                plt.clf()
                break
        # When everything is done, release the capture
        np.savetxt
        plt.title('Overall Emotion Ratio')
        plt.savefig(output_plot)
        video_capture.release()
        cv2.destroyAllWindows()
        plt.clf()
        plt.close()
        # cmd = 'eog '+ output_plot
        # os.system(cmd)
        # img = mpimg.imread(output_plot)
        # plt.imshow(img)
        # plt.show()
        # cv2.imshow('Emotion Plot', output_plot)


    def classify_faces_recorded_movie(self, file_path=0, write_imgs=False, output_plot='../images/movie.png'):
        # self.capture_duration = duration
        start_time = time.time()
        video_capture = cv2.VideoCapture(file_path)
        # self.results_df
        self.total_df_probas = []
        self.total_df_predict = []
        # plt.ion()
        # self.results_df
        self.ret = True
        while self.ret == True:
            # Capture frame-by-frame
            self.ret, self.frame = video_capture.read()
            print(self.ret)
            if not self.ret:
                break 
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f'Found {len(faces)} faces')
            plt.clf()
            if len(faces)>0:
                self.temp_df_probas = []
                self.temp_df_predict = []
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    sub_face = self.frame[y:y+h, x:x+w]
                    if write_imgs:
                        face_file_name = "faces/face_" + str(y) + ".jpg"
                        cv2.imwrite(face_file_name, sub_face)
                    gray_image = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                    sb2 = cv2.resize(gray_image, (48, 48)) 
                    sb3 = np.expand_dims(sb2, axis=3) 
                    sb4 = np.array([sb3])

                    self.test_pred_y = self.best_model.predict_classes(sb4)
                    self.test_pred_proba = self.best_model.predict_proba(sb4)
                    print(self.test_pred_y)
                    print(self.test_pred_proba)
                    print(self.emo_dict[self.test_pred_y[0]])

                    self.temp_df_probas.append(self.test_pred_proba)
                    self.temp_df_predict.append(self.test_pred_y[0])
                self.total_df_probas.append(np.array(self.temp_df_probas).mean(axis=0)[0])
                mode = Counter(self.temp_df_predict).most_common(1)
                self.total_df_predict.append(mode[0][0])
            else:
                self.total_df_probas.append(np.array([0, 0, 0, 0, 0, 0]))
                self.total_df_predict.append(np.NaN)
                self.test_pred_y =[99]
            self.means_to_plot = np.array(self.total_df_probas).mean(0)
            # plt.bar(self.x_range, self.means_to_plot)
            # # sleep(.5)
            # # plt.pause(0.5)
            # plt.title(self.emo_dict[self.test_pred_y[0]])
            # plt.xticks(range(6), list(self.emo_dict.values()))
            # plt.ylim(0,1)
            # cv2.imshow('Video', frame)
            # plt.draw()
            # plt.pause(.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                plt.clf()
                break

        # When everything is done, release the capture
        self.means_to_plot = np.array(self.total_df_probas).mean(0)
        plt.bar(self.x_range, self.means_to_plot)
        plt.xticks(range(6), list(self.emo_dict.values()))
        plt.title('Overall Emotion Ratio')
        plt.savefig(output_plot)
        video_capture.release()
        cv2.destroyAllWindows()
        plt.clf()
        plt.close()
        # cmd = 'eog '+ output_plot
        # os.system(cmd)


if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    # home = '/home/ubuntu/efc/src/'
    cv2_path = '/home/danny/anaconda3/lib/python3.6/site-packages/cv2/data/'
    bestmodelfilepath = home + 'CNN_cont.hdf5'
    efp = EmotionFacePredictor(home, cv2_path, bestmodelfilepath)
    efp.run_setup()
    # efp.classify_faces_image('./faces/face_174.jpg')
    efp.classify_faces_video()
