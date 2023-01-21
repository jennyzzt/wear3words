import os
from flask import Flask, render_template, request, url_for, send_from_directory
from markupsafe import Markup

from calculate import measurements_to_words, image_to_measurements


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./uploaded_images"
app.config["IMAGE_PROCESSED"] = "./uploaded_images/processed"


@app.route('/uploaded_images/processed/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config["IMAGE_PROCESSED"], filename, as_attachment=True)

@app.route("/", methods=["GET", "POST"])
def get_measurements():
    words3disp = ""
    imgestdisp = ""
    if request.method == "POST":
        input_image_success = False
        if 'front_image' in request.files and 'side_image' in request.files and 'bodylen' in request.form:
            image = request.files['front_image']
            image_side = request.files['side_image']
            if image and image_side:
                # estimate measurements from input image
                input_image_success = True
                filepath = os.path.join(app.config["IMAGE_UPLOADS"], 'person_front.png')
                image.save(filepath)
                filepath_side = os.path.join(app.config["IMAGE_UPLOADS"], 'person_side.png')
                image_side.save(filepath_side)
                bodylen = float(request.form['bodylen'])
                body_msmts = image_to_measurements(filepath, filepath_side, bodylen)
                # format html to be shown
                msmt_names = ['chest', 'waist', 'hip']
                imgestdisp = ""
                for name in msmt_names:
                    imgestdisp += f'<pre>{name} keypoints:</pre>'
                    imgestdisp += '<div class="text-center">'
                    imgestdisp += f'<img class="img-fluid img-thumbnail" src="./uploaded_images/processed/{name}_front_est.png" style="width: auto; max-height: 200px;"/>'
                    imgestdisp += f'<img class="img-fluid img-thumbnail" src="./uploaded_images/processed/{name}_side_est.png" style="width: auto; max-height: 200px;"/>'
                    imgestdisp += '</div><br>'
                imgestdisp = Markup(imgestdisp)
        if not input_image_success:
            # get input body measurements
            body_msmts = {}
            for key in request.form:
                if key == 'bodylen':
                    continue
                try:
                    body_msmts[key] = float(request.form[key])
                except:
                    # not enough input
                    words3disp = Markup("<div class='alert alert-warning'><b>!</b> Please input all values or choose <b>Interpolate</b> / <b>Photo Input</b></div>")
                    return render_template('input.html', value={
                        'words3disp': words3disp,
                        'imgestdisp': imgestdisp,
                    })
        # calculate the three words
        words3 = measurements_to_words(body_msmts)
        words3 = f'///{".".join(words3)}'
        # html pretty print for measurements
        msmts_pprt = ""
        for k, v in body_msmts.items():
            msmts_pprt += f'{k}: {v}\n'
        words3disp = Markup(f"<h3>{words3}</h3><pre>{msmts_pprt}</pre>")
    return render_template('input.html', value={
        'words3disp': words3disp,
        'imgestdisp': imgestdisp,
    })
