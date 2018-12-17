import os
import glob
import numpy as np 
import matplotlib.pyplot as plt 
from FaceDetector_v9 import EmotionFacePredictor
plt.rcParams.update({'font.size': 18})

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
    # my_movies_path = '/mnt/b59a3b67-b6fe-455f-a4cf-e238a0feee58/My_Movies/'
    # my_movies=glob.glob(my_movies_path+'*m4v')
    # my_movies = ["Hp1 Sorcerers Stone.m4v", "Hp7 Deathly Hallows Part 2.m4v", "300.m4v", "Supertroopers.m4v", "Zoolander.m4v", "Inside Out.m4v", "Simpsons.m4v", "V For Vendetta.mkv", "Spaceballs.m4v", "The_Incredibles.m4v", "Pineapple Exp.m4v", "Robinhood: Men in Tights.mkv", "Wreck It Ralph.mkv"]
    movie_matrices = ['HP1', 'HP8', '300', 'Simpsons']
    movie_titles = [ 'Harry Potter and \nthe Sorcerers Stone' , 'Harry Potter and \nthe Deathly Hallows Part II', '300', 'The Simpsons Movie']    
    # ax = plt.subplot(1,1,1)
    fig = plt.figure(figsize=(50, 50))
    ax1 = fig.add_subplot(111)    # The big subplot
    # ax_overall.set_ylabel('common ylabel')
    plt.suptitle('Movie Comparison')
    for i, mv, title in zip(range(4), movie_matrices, movie_titles):
        proba_path = '../images/'+ mv + '_probas.txt'
        subplot = i+1
        ax = plt.subplot(2,2,subplot)
        ax.set_title(title)
        temp_array = np.loadtxt(proba_path)
        clipped_array = temp_array[~np.all(temp_array==0, axis=1)]
        means = clipped_array.mean(0)
        ax.bar(efp.x_range, means, color=efp.emo_colors)
        ax.set_ylim([0, .5])
        print(title)
        print(f'Means sum to : {np.sum(means)}')
        if subplot >2:
            print('adding xticks')
            ax.set_xticks(np.arange(6))
            labels = list(efp.emo_dict.values())[:-1]
            ax.set_xticklabels(labels)
            print(labels)
        else:
            ax.set_xticks([])
    # plt.ylabel('Emotion %')
    fig.text(0.06, 00.5, 'Emotion %',fontsize=20, ha='center', va='center', rotation='vertical')
    # ax1.set_ylabel('Emotion %')
    plt.savefig('../images/Movie_Comparison.png')
    plt.show()
    plt.close()
