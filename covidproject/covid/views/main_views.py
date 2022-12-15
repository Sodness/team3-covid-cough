import time

from flask import Blueprint, url_for, render_template, flash, request
import os

import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import torchvision
from torchvision import datasets, models, transforms

import numpy as np

bp = Blueprint('main',__name__,url_prefix='/')

model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model.load_state_dict(torch.load('C:/team3/covidproject/covid/models/model_dict.pth'), strict=False)
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

        input_audio_path = 'audio.wav'
        x, sr = librosa.load(input_audio_path)
        fig, ax = plt.subplots(figsize=(15, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(x, hop_length=160)), ref=np.max)
        tmp = librosa.display.specshow(D, hop_length=160)
        plt.savefig('mel.jpg')

        image = Image.open('./02227.jpg')
        image = transforms_test(image).unsqueeze(0).to('cpu')
        with torch.no_grad():
            outputs = model(image)

            _, preds = torch.max(outputs, 1)
            print(preds[0])

        return render_template('Recording.html', request="POST")
    else:
        return render_template("Recording.html")



