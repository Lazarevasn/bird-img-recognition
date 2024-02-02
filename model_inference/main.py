"Module used for reading image path from the command line"
import sys
from modules.model_initialization import model_initializer
from modules.image_preprocessing import image_preprocessor

birds_dict = {
        0: 'Гусь',
        1: 'Индюки',
        2: 'Курица',
        3: 'Петух',
        4: 'Страус',
        5: 'Утка',
        6: 'Цыпленок'
    }

if __name__=="__main__":
    model = model_initializer.BirdRecognitionModel('bird_recognition.h5')
    img_processor = image_preprocessor.ImagePreprocessor()
    results = model.make_predictions(img_array = img_processor.preprocess_image(sys.argv[1]))
    print(f"Класс: {birds_dict[results[1]]}, Результаты предсказания: {results[0]}")
