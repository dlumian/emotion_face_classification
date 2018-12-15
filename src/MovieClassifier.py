import os
import glob
from FaceDetector_v8 import EmotionFacePredictor


if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    # home = '/home/ubuntu/efc/src/'
    cv2_path = '/home/danny/anaconda3/lib/python3.6/site-packages/cv2/data/'
    bestmodelfilepath = home + 'CNN_cont.hdf5'
    efp = EmotionFacePredictor(home, cv2_path, bestmodelfilepath)
    efp.run_setup()
    my_movies_path = '/mnt/b59a3b67-b6fe-455f-a4cf-e238a0feee58/My_Movies/'
    # my_movies=glob.glob(my_movies_path+'*m4v')
    my_movies = ["Hp1 Sorcerers Stone.m4v", "Hp7 Deathly Hallows Part 2.m4v", "300.m4v", "Supertroopers.m4v", "Zoolander.m4v"]
    movie_titles = ['HP1', 'HP8', '300', 'Supertroopers', 'Zoolander']
    # test_movie = my_movies_path+'300.m4v'
    # test_movie = '../stims/test_vids/test.webm'
    for mv, title in zip(my_movies, movie_titles):
        efp.classify_faces_recorded_movie(my_movies_path+mv, write_imgs=False, output_plot='../images/'+title+'.png')
    # for mv in my_movies:
    #     efp.classify_faces_recorded_movie(mv)
