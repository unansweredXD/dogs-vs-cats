import os
import pathlib

# Указываем путь к директории
directory = "PetImages/Cat"

# Создаем пустой список
files = []

# Добавляем файлы в список
files += os.listdir(directory)

# На Windows путь будет что-то вроде:
# D:\\user\\images
os.mkdir('train_all')

path = pathlib.Path("PetImages/Dog")
for i, path in enumerate(path.glob('*.jpg')):
    new_name = 'dog_' + str(i) + '.jpg'
    path.rename('train_all/' + new_name)

path = pathlib.Path("PetImages/Cat")
for i, path in enumerate(path.glob('*.jpg')):
    new_name = 'cat_' + str(i) + '.jpg'
    path.rename('train_all/' + new_name)
