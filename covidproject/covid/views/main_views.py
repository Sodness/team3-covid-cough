from flask import Blueprint, url_for, render_template, flash, request
import os

bp = Blueprint('main',__name__,url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route("/process_wav", methods=['GET', 'POST'])
def process_wav():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        if os.path.isfile('./file.wav'):
            print("./file.wav exists")
        return render_template('Recording.html', request="POST")
    else:
        return render_template("Recording.html")
