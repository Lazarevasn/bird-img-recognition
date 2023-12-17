# BirdImageRecognition (TensorFlow)

# Инференс модели:
## Подготовка:
1. Скачать файл _model\_inference/bird\_predictor.py_.
2. Установить необходимые библиотеки из _model\_inference/requirements.txt_.

Файл с весами и конфигурацией обученной модели _bird_recognition.h5_ будет установлен автоматически при первом использовании bird_predictor.py. 
Готовое виртуальное окружение вместе с файлами _bird_predictor.py_ и _bird_recognition.h5_ можно скачать [отсюда](https://drive.google.com/file/d/1Y-jO4jANrROTUYds51mj3tPZHXAALCZz/view?usp=sharing).

## Запуск:
Для запуска модели на инференс необходимо запустить файл _bird_predictor.py_ и передать в качестве параметра путь к изображению на распознавание.
Пример: 

**.\birdenv\Scripts\python.exe bird_predictor.py .\chicken.jpg**

# Работа с JupyterLab с помощью Docker

## Подготовка и запуск

1. Скачать файлы из директории _jnotebook_.
2. Находясь в директории со скачанными файлами, использовать команду **docker-compose build**.
3. По завершении процесса создания образа использовать команду **docker-compose up**.
4. После запуска проекта открыть в браузере одну из ссылок из консоли.