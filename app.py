import os
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import requests
from markupsafe import Markup

from calculate import measurements_to_words, image_to_measurements
from generate_video import words_to_video


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./uploaded_images"
app.config["IMAGE_PROCESSED"] = "./uploaded_images/processed"


@app.route('/uploaded_images/processed/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config["IMAGE_PROCESSED"], filename, as_attachment=True)

@app.route('/video_time')
def video_time():
    word0 = request.args.get('word0')
    word1 = request.args.get('word1')
    word2 = request.args.get('word2')
    words = [word0, word1, word2]
    print('getting video')
    videolink = words_to_video(words)
    # videolink = "https://replicate.delivery/pbxt/vOIM4etmY0TYQi3MqSrTeKnvSNCNOdhO2wRSYKGe0taYBctgA/out.mp4"  # for debugging
    # use livepeer to encode video
    livepeer_api_key = os.environ.get('LIVEPEER_API_TOKEN')
    response = requests.post(
        url='https://livepeer.studio/api/asset/import',
        data='{"url":"' + videolink + '","name":"w3w user"}',
        headers={
            'Authorization': f'Bearer {livepeer_api_key}',
            'Content-Type': 'application/json'
        }
    )
    playbackid = response.json()['asset']['playbackId']
    # playbackid = '3ce5nbcrrq3g5bfb'  # for debugging
    videolink = f'https://lvpr.tv?v={playbackid}'
    return jsonify({'videolink': videolink})

@app.route("/", methods=["GET", "POST"])
def get_measurements():
    words3disp = ""
    imgestdisp = ""
    words3 = [""] * 3
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
        words3_print = f'///{".".join(words3)}'
        # html pretty print for measurements
        msmts_pprt = ""
        for k, v in body_msmts.items():
            msmts_pprt += f'{k}: {v}\n'
        words3disp = Markup(f"""
            <h3 style="display: inline;">{words3_print}</h3>&nbsp
            <input id="genvideo" class="btn-rainbow" type="button" value="Livepeer this!""/>
            <pre>{msmts_pprt}</pre>
        """)
    return render_template('input.html', value={
        'words3disp': words3disp,
        'imgestdisp': imgestdisp,
        'words3': words3,
    })


if __name__ == '__main__':
   app.run()
