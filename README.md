# BirdImageRecognition (TensorFlow)

## Подготовка:
1. Скачать файл _bird_predictor.py_.
2. Установить необходимые библиотеки из _requirements.txt_.

Файл с весами и конфигурацией обученной модели _bird_recognition.h5_ будет установлен автоматически при первом использовании bird_predictor.py. 
Готовое виртуальное окружение вместе с файлами _bird_predictor.py_ и _bird_recognition.h5_ можно скачать [отсюда](https://drive.google.com/file/d/1Y-jO4jANrROTUYds51mj3tPZHXAALCZz/view?usp=sharing).

## Инференс:
Для запуска модели на инференс необходимо запустить файл _bird_predictor.py_ и передать в качестве параметра путь к изображению на распознавание.
Пример: 

**.\birdenv\Scripts\python.exe bird_predictor.py .\chicken.jpg**
