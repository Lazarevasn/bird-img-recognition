# BirdImageRecognition (TensorFlow)

## Инференс модели

### Подготовка

1. Скачать и распаковать архив с файлами репозитория в локальный каталог.
2. Скачать и установить [Python 3.11](https://www.python.org/downloads/).
3. В локальной папке с файлами репозитория, установить и активировать виртуальное окружение:
   1. Открыть командную строку или терминал.
   2. Перейти в нужный локальный каталог.
   3. Ввести команду **python venv birdenv**. (birdenv - имя окружения, может быть любым)
   4. Активировать окружение, использовав команду **birdenv/Scripts/activate**.
4. В виртуальном окружении установить необходимые библиотеки из файла _model\_inference/requirements.txt_, использовав команду **pip install -r model\_inference/requirements.txt**.

Файл с весами и конфигурацией обученной модели _bird_recognition.h5_ будет установлен автоматически при первом использовании bird_predictor.py.
Датасеты, использованные при обучении и тестировании модели, можно скачать с [Google Drive](https://drive.google.com/drive/folders/1ZDlgOtVB-Jdkqt-cQrEdyl-QMCJwF2kq?usp=sharing).

### Запуск

Для запуска модели на инференс необходимо запустить файл _model\_inference/bird_predictor.py_ и передать в качестве параметра путь к изображению на распознавание.
Для этого, находясь в активированном окружении, необходимо ввести команду:

**python bird_predictor.py chicken.jpg**

Вместо chicken.jpg укажите путь к файлу с изображением.

## Работа с JupyterLab с помощью Docker

### Подготовка и запуск

1. Скачать и распаковать архив с файлами репозитория в локальный каталог.
2. Открыть командную строку или терминал и перейти в подпапку /jnotebook.
3. Создать образ, используя команду **docker-compose build**.
4. По завершении процесса создания образа использовать команду **docker-compose up**.
5. После появления информации в консоли о завершении запуска проекта открыть в браузере одну из ссылок из консоли.
