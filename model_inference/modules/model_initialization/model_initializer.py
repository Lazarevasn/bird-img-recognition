"Module used to define whether the file with model weights has already been downloaded"
import os
import numpy as np
from keras.models import load_model, Model
from .utilities import download_file_from_url

class BirdRecognitionModel:
    '''
    Describes the ML model used for predicting the type of bird in the image.

      Attributes:
        __weights_path(str): the name of the file which contains the model weights.
        __model(Model): the model itself

      Methods:
      __download_model_weights(self) -> None: Downloads the model weights from a file stored 
                                              remotely into a local file.
      __initialize_model(self) -> Model: Initializes model loading its weights from a local file.
      make_predictions(self, img_array: ndarray) -> tuple[np.float32, np.int64]: Makes predictions 
                                                                about the type of bird in the image.
    '''

    def __init__(self, weights_path: str) -> None:
        '''
        Initializes the model with a set path to the file with the model weights.
          Parameters:
            weights_path(str): the path to the local file which contains the model weights
          Returns:
            None 
        '''
        self.__weights_path = weights_path
        self.__model = self.__initialize_model()

    def __download_model_weights(self) -> None:
        '''
        Downloads the model weights from a file stored remotely into a local file.
          Parameters:
            None
          Returns:
            None
        '''
        file_url = 'https://drive.google.com/uc?id=1-bWaePLw8owneHJnK2US7jcYnG5gaI5j'
        download_file_from_url(file_url, self.__weights_path)

    def __initialize_model(self) -> Model:
        '''
        Initializes the model loading its weights from a local file. If the file hasn't yet been 
        downloaded, calls the __download_model_weights() function first and then loads model weights
        from the downloaded file.
          Parameters:
            None
          Returns:
            loaded_model(tf.keras.models.Model): model loaded using the downloaded model weights
        '''
        if not os.path.isfile(self.__weights_path):
            self.__download_model_weights()
        loaded_model = load_model(self.__weights_path)
        return loaded_model

    def make_predictions(self, img_array: np.ndarray) -> tuple[np.float32, np.int64]:
        '''
        Makes predictions about the type of bird is in the image using the loaded model.
        Parameters:
          img_array(np.ndarray): preprocessed image in the form of a numpy array
        Returns:
          score(np.float32): the probability of the predicted bird kind being correct
          kind(np.int64): the number corresponding to the predicted bird kind and one of the keys 
                          in birds_dict dictionary
        '''
        predictions = self.__model.predict(img_array)
        score = np.max(predictions)
        kind = np.where(predictions==np.max(predictions))[1]
        kind = kind[0]
        return score, kind
    