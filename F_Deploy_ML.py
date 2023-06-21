import os
from flask import Flask
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model_serviks = tf.keras.models.load_model('model_serviks.h5')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
nama_file = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('Deploy_ML/upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    global nama_file
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        nama_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(nama_file)
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded, start to predicting')
        return redirect(url_for('HasilPrediksi'))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def prediksi_gambar(filename):
    gambar_prediksi_1 = filename
    gambar_inferensi = image.load_img(gambar_prediksi_1, target_size=(224, 224))
    img_array = image.img_to_array(gambar_inferensi)
    img_batch = np.expand_dims(img_array, axis=0)
    hasil_prediksi = model_serviks.predict(img_batch)[0][0]
    return hasil_prediksi

@app.route('/HasilPrediksi')
def HasilPrediksi():
    file_name = nama_file
    hasil_prediksi = prediksi_gambar(file_name)
    tampilan = 'Positif'
    if hasil_prediksi<1:
        tampilan = "Negatif"
    else:
        tampilan = "Positif"
    return render_template('Deploy_ML/prediksi.html', gambar = file_name, tampilan=tampilan)

if __name__ == "__main__":
    app.run()