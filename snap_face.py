import os
import glob
import json

from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, send_file
from werkzeug.utils import secure_filename

import core

base_path = os.getcwd()
upload_folder = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def clear_uploads():
    import shutil
    import os.path
    shutil.rmtree(upload_folder, ignore_errors=True)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        thefile = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if thefile.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if thefile and allowed_file(thefile.filename):
            original_file_content = thefile.read()
            thefile.close()
            img_resized_str = core.resize_image_string(original_file_content, height=240)

            filename = secure_filename(thefile.filename)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
                f.write(img_resized_str)

            return redirect(url_for('effect', filename=filename))
    return render_template('upload_page.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/photos/')
def photos():
    photo_paths = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.jpg'))
    filenames = [os.path.split(p)[-1] for p in photo_paths]
    return render_template('photos.html', filenames=filenames)


@app.route('/features/<filename>')
def face_features(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    theface = core.Face(img_path)
    image_binary = core.convert_image(theface.mark_features())
    data_uri = image_binary.encode('base64').replace('\n', '')
    image_source = "data:image/png;base64,{0}".format(data_uri)
    return render_template('result.html', image_source=image_source)


@app.route('/effects/<filename>', methods=['GET', 'POST'])
def effect(filename):
    image_source = url_for('uploaded_file', filename=filename)

    effect_list = ['sg_red.png', 'sg_black.png', 'sg_aviator_navy.png',
                   'sg_pink.png', 'sg_aviator_white.png']
    effects = [url_for('static', filename='img/glasses/' + ef) for ef in effect_list]

    if request.method == 'POST':
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        theface = core.Face(img_path)
        glass_path = "." + request.form['glass']
        theglasses = core.Glasses(glass_path)

        img = theface.wear_glass(theglasses.get_image())
        image_binary = core.convert_image(img)
        data_uri = image_binary.encode('base64').replace('\n', '')
        image_source = "data:image/png;base64,{0}".format(data_uri)
        return render_template('result.html', image_source=image_source)

    return render_template('effect.html', image_source=image_source, effects=effects)


if __name__ == '__main__':
    # clear_uploads()
    app.run()

    # TODO database for uploads
    # TODO glass rotation
    # TODO glass dilation
