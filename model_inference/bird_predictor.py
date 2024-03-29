# -*- coding: utf-8 -*-
import os,sys
import gdown
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

birds_dict = {
        0: 'Гусь',
        1: 'Индюки',
        2: 'Курица',
        3: 'Петух',
        4: 'Страус',
        5: 'Утка',
        6: 'Цыпленок'
    }

def model_upload():
  output = 'bird_recognition.h5'
  if not os.path.isfile(output):
    file_id = '1-bWaePLw8owneHJnK2US7jcYnG5gaI5j'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
  model = load_model(output)
  return model


def preprocess_img(image_path):
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)
  return img_array

def predict(model,img_array):
  predictions = model.predict(img_array)
  score = np.max(predictions)
  kind = np.where(predictions==np.max(predictions))[1]
  kind=kind[0]
  return score,kind


if __name__=="__main__":
  image_path=sys.argv[1]
  results=predict(model_upload, preprocess_img(image_path))
  print(f"Класс: {birds_dict[results[1]]}, Результаты предсказания: {results[0]}")