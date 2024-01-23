import os,sys
import requests
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
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

def download_file_from_url(url:str, destination_path:str) -> None:
    response = requests.get(url, stream=True)
    
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

class Model:

  def __init__(self, weights_path:str) -> None:
    self.__weights_path = weights_path
    self.__model = self.__initialize_model()
  
  def __download_model_weights(self) -> None:
    if not os.path.isfile(self.__weights_path):
      file_url = 'https://drive.google.com/uc?id=1-bWaePLw8owneHJnK2US7jcYnG5gaI5j'
      download_file_from_url(file_url, self.__weights_path)

  def __initialize_model(self) -> Model:
    self.__download_model_weights()
    return load_model(self.__weights_path)

  def make_predictions(self, img_array:np.ndarray) -> tuple[np.float32, np.int64]:
    predictions = self.__model.predict(img_array)
    score = np.max(predictions)
    kind = np.where(predictions==np.max(predictions))[1]
    kind=kind[0]
    return score,kind
  
class Image:
   
  def __init__(self, image_path:str) -> None:
    self.__image_path = image_path

  def preprocess_image(self) -> np.ndarray:
    img = image.load_img(self.__image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

if __name__=="__main__":
  model = Model('bird_recognition.h5')
  bird_image = Image(sys.argv[1])
  
  results = model.make_predictions(img_array=bird_image.preprocess_image())
  print(f"Класс: {birds_dict[results[1]]}, Результаты предсказания: {results[0]}")