import os
import glob
import numpy as np 
import matplotlib.pyplot as plt 
from FaceDetector_v9 import EmotionFacePredictor


if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/emotion_face_classification/src/'
    # home = '/home/ubuntu/efc/src/'
    cv2_path = '/home/danny/anaconda3/lib/python3.6/site-packages/cv2/data/'
    bestmodelfilepath = home + 'CNN_cont.hdf5'
    efp = EmotionFacePredictor(home, cv2_path, bestmodelfilepath)
    efp.run_setup()
    # 15s webcam test
    # efp.classify_faces_video(file_path=0, 
    #                             duration=5, 
    #                             write_imgs=False, 
    #                             output_name='test', 
    #                             show_plots=True,
    #                             show_final_plot=True)
    my_movies_path = '/mnt/b59a3b67-b6fe-455f-a4cf-e238a0feee58/My_Movies/'
    # my_movies=glob.glob(my_movies_path+'*m4v')
    my_movies = ["Hp1 Sorcerers Stone.m4v", "Hp7 Deathly Hallows Part 2.m4v", "300.m4v", "Supertroopers.m4v", "Zoolander.m4v", "Inside Out.m4v", "Simpsons.m4v", "V For Vendetta.mkv"]
    movie_titles = ['HP1', 'HP8', '300', 'Supertroopers', 'Zoolander', 'InsideOut', 'Simpsons', 'Vendetta']    
    # test_movie = my_movies_path+'300.m4v'
    # test_movie = '../stims/test_vids/test_2.webm'
    # efp.classify_faces_video(file_path=test_movie, 
    #                             duration=0, 
    #                             write_imgs=False, 
    #                             output_name='test_vid2', 
    #                             show_plots=True,
    #                             show_final_plot=True) 

    for mv, title in zip(my_movies, movie_titles):
        if not os.path.isfile('../images/' + title + '.png'):
            efp.classify_faces_video(file_path=my_movies_path+mv, 
                                        duration=0, 
                                        write_imgs=False, 
                                        output_name=title, 
                                        show_plots=False,
                                        show_final_plot=False)
        if not os.path.isfile('../images/' + title + '_2.png'):
            proba_path = '../images/'+ title + '_probas.txt'
            if os.path.isfile(proba_path):
                temp_array = np.loadtxt(proba_path)
                clipped_array = temp_array[~np.all(temp_array==0, axis=1)]
                means = clipped_array.mean(0)
                print(title)
                print(f'Means sum to : {np.sum(means)}')
                plt.bar(efp.x_range, means, color=efp.emo_colors)
                plt.title(title)
                plt.xticks(range(6), list(efp.emo_dict.values()))
                plt.ylim(0, .5)
                plt.savefig('../images/'+title+'_2.png')
                plt.close()

        # for mv, title in zip(my_movies, movie_titles):
    #     efp.classify_faces_video(my_movies_path+mv, write_imgs=False, output_plot='../images/'+title+'.png')
    # for mv in my_movies:
    #     efp.classify_faces_recorded_movie(mv)
