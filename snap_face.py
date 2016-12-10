import os
import glob
import json

from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, abort
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

with open('accessories.json') as f:
    accessory_collection = json.load(f)

glasses_collection = [ac for ac in accessory_collection if ac['kind'] == 'glasses']
hat_collection = [ac for ac in accessory_collection if ac['kind'] == 'hat']


def clear_uploads():
    import shutil
    import os.path
    shutil.rmtree(upload_folder, ignore_errors=True)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_accessory_object(pk):
    queryset = [ac for ac in accessory_collection if ac['pk'] == pk]
    if len(queryset) == 0:
        return None
    accessory_item = queryset[0]
    accessory_path = os.path.join(app.config['STATIC_FOLDER'], 'img/accessory', pk + '.png')
    accessory_obj = core.Thing(accessory_path, float(accessory_item['cx']), float(accessory_item['cy']),
                               accessory_item['kind'], float(accessory_item.get('scale_factor', 1)))
    return accessory_obj


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
            img_resized_str = core.resize_image_string(original_file_content, height=480)

            filename = secure_filename(thefile.filename)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
                f.write(img_resized_str)

            return redirect(url_for('wear_accessory', filename=filename))
    return render_template('upload.html')


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


@app.route('/accessories/<filename>', methods=['GET', 'POST'])
def wear_accessory(filename):
    image_source = url_for('uploaded_file', filename=filename)

    if request.method == 'POST':
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        theface = core.Face(img_path)

        pks = request.form.getlist('accessory')

        accessory_objs = [get_accessory_object(pk) for pk in pks]

        kinds = [acc.kind for acc in accessory_objs]
        if len(kinds) != len(set(kinds)):
            abort(403, 'You cannot wear multiple accessories of same kind')

        # TODO wear multiple hat and glass at the same time
        img = theface.wear_accessories(accessory_objs)
        image_binary = core.convert_image(img)
        data_uri = image_binary.encode('base64').replace('\n', '')
        image_source = "data:image/png;base64,{0}".format(data_uri)
        return render_template('result.html', image_source=image_source)

    return render_template('accessory.html', image_source=image_source, accessories=accessory_collection)


if __name__ == '__main__':
    # clear_uploads()
    app.run()

    # TODO database for uploads
