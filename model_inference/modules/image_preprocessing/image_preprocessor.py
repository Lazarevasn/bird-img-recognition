import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

class ImagePreprocessor:
    '''
    Describes an image preprocessor object.

      Attributes:
        None

      Methods:
        preprocess_image(self, image_path:str): Preprocesses an image from a received path 
                                                converting it into a numpy array.
    '''
    def __init__(self) -> None:
        '''
        Initializes an image preprocessor.
          Parameters:
            None
          Returns: 
            None
        '''

    def preprocess_image(self, image_path: str) -> np.ndarray:
        '''
        Preprocesses an image from a received path converting it into a numpy array.
          Parameters:
            image_path(str): path to the local file containing an image
          Returns:
            preprocessed_img(np.ndarray): chosen image in the form of a numpy array 
        '''
        img = image.load_img(image_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_array)
        return preprocessed_img
