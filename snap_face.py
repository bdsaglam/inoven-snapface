import os
import json

from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, send_file
from werkzeug.utils import secure_filename

import core

base_path = os.getcwd()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


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
            filename = secure_filename(thefile.filename)
            thefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload_page.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/features/<filename>')
def face_features(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    theface = core.Face(img_path)
    image_binary = core.convert_image(theface.plot_features())
    data_uri = image_binary.encode('base64').replace('\n', '')
    image_source = "data:image/png;base64,{0}".format(data_uri)
    return render_template('features.html', image_source=image_source)


@app.route('/photos/<filename>', methods=['GET', 'POST'])
def effect(filename):
    image_source = url_for('uploaded_file', filename=filename)

    effect_list = ['sg_red.png', 'sg_black.png', 'sg_aviator_navy.png']
    effects = [url_for('static', filename='img/glasses/' + ef) for ef in effect_list]

    if request.method == 'POST':
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        theface = core.Face(img_path)
        glass_path = "." + request.form['glass']
        theglasses = core.Glasses(glass_path)
        print img_path, glass_path

        img = theface.wear_glass(theglasses.get_image())

        image_binary = core.convert_image(img)
        data_uri = image_binary.encode('base64').replace('\n', '')
        image_source = "data:image/png;base64,{0}".format(data_uri)
        return render_template('features.html', image_source=image_source)

    return render_template('effect.html', image_source=image_source, effects=effects)


if __name__ == '__main__':
    app.run()
