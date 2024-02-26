import os
import pathlib


def generate_dataset():
    directory = "PetImages/Cat"

    files = []

    files += os.listdir(directory)

    os.mkdir('dataset')

    path = pathlib.Path("PetImages/Dog")
    for i, path in enumerate(path.glob('*.jpg')):
        new_name = 'dog_' + str(i) + '.jpg'
        path.rename('dataset/' + new_name)

    path = pathlib.Path("PetImages/Cat")
    for i, path in enumerate(path.glob('*.jpg')):
        new_name = 'cat_' + str(i) + '.jpg'
        path.rename('dataset/' + new_name)
