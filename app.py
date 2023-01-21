import os
from flask import Flask, render_template, request, url_for, send_from_directory


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./uploaded_images"


@app.route("/", methods=["GET", "POST"])
def get_measurements():
    return render_template('input.html')
