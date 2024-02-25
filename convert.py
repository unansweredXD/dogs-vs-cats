import os

# Указываем путь к директории
directory = "PetImages/Cat"

# Создаем пустой список
files = []

# Добавляем файлы в список
files += os.listdir(directory)

import pathlib

# На Windows путь будет что-то вроде:
# D:\\user\\images
path = pathlib.Path("PetImages/Dog")
for i, path in enumerate(path.glob('*.jpg')):
    new_name = 'dog_' + str(i) + '.jpg'
    path.rename(new_name)