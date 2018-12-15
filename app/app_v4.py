import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pickle
from FaceDetector_v7 import EmotionFacePredictor
import json
from werkzeug.utils import secure_filename
import tensorflow as tf 
global graph,model

graph = tf.get_default_graph()
plt.ioff()

app = Flask(__name__)

home = os.getcwd()
# home = '/home/ubuntu/efc/src/'
cv2_path = './'
bestmodelfilepath = './CNN_cont.hdf5'
efp = EmotionFacePredictor(home, cv2_path, bestmodelfilepath)
efp.run_setup()

UPLOAD_FOLDER = 'static/images/'
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
# prevent cached responses
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict',
                                    filename=filename))
    return render_template('form/index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    response = request.args.get('filename')
    image = os.path.join(app.config['UPLOAD_FOLDER'], response)
    print(image)
    with graph.as_default():
        results = efp.classify_faces_image(image)
    print("RESULTS HERE:")
    print(results)
    if not results:
        print("NOTHING TO SEE HERE")
        return render_template('form/no_face_found.html', response=image)
    # else:
    print("Found some faces!!")
    top_emotions = [efp.emo_list[x[0]] for x in results[1]]
    return render_template('form/predict.html', 
                            orig_img=image, 
                            faces = results[0],
                            top_emos = top_emotions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
