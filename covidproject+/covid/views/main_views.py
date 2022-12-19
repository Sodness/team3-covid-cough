from flask import Blueprint, url_for, render_template, flash, request,redirect,escape
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings('ignore')

bp = Blueprint('main',__name__,url_prefix='/')

model = models.resnet34(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model.load_state_dict(torch.load("C:\\Users\\user\\Desktop\\KHW\\covidproject\\covid\\models\\48-model_dict.pth"), strict=False)
model.eval()
print('모델 로드 완료')
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = model.to('cpu')

@bp.route('/')
def index():
    return render_template('home.html')

@bp.route("/process_wav", methods=['GET', 'POST'])
def process_wav():
    print('Start')
    if request.method == "POST":
        print('POST')
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        print('11111111111')
        return render_template("Recording.html", methods='post')
    else:
        print('GET')
        return render_template("Recording.html")

@bp.route('/result')
def result():
    input_audio_path = 'audio.wav'
    x, sr = librosa.load(input_audio_path)
    fig, ax = plt.subplots(figsize=(15, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x, hop_length=160)), ref=np.max)
    tmp = librosa.display.specshow(D, hop_length=160)
    plt.savefig('mel.jpg')
    if os.path.isfile('./file.wav'):
        print("./file.wav exists")

    image = Image.open('mel.jpg')
    image = transforms_test(image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        outputs = model(image)

        _, preds = torch.max(outputs, 1)
        pred = preds[0].item()
        print(pred)
    return render_template('result.html', pred=pred)