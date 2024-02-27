import glob
import os
import pathlib
import zipfile

from sklearn.model_selection import train_test_split


class DatasetLoader:
    def __init__(self):
        self.train_files = None
        self.test_files = None
        self.dataset_path = f'dataset\\'
        self.train_dir = f'dataset\\data\\'
        self.dataset_archive_path = f'microsoft-catsvsdogs-dataset.zip'

        self.dataset_folder = os.listdir(self.dataset_path)
        self.load_images()

    @staticmethod
    def split_dataset():
        directory = 'dataset\\PetImages\\Cat'

        files = []

        files += os.listdir(directory)

        os.mkdir('dataset\\data')

        path = pathlib.Path('dataset\\PetImages\\Dog')
        for i, path in enumerate(path.glob('*.jpg')):
            new_name = 'dog_' + str(i) + '.jpg'
            path.rename('dataset\\data\\' + new_name)

        path = pathlib.Path('dataset\\PetImages\\Cat')
        for i, path in enumerate(path.glob('*.jpg')):
            new_name = 'cat_' + str(i) + '.jpg'
            path.rename('dataset\\data\\' + new_name)

    def extract_dataset(self):
        try:
            with zipfile.ZipFile(self.dataset_archive_path, 'r') as file:
                file.extractall(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError('Проверьте, что архив с датасетом находится в корне проекта!')

    def split_train(self, test_size=0.2, val_size=0.4):
        train_list, test_list = train_test_split(self.train_files, test_size=test_size, train_size=1 - test_size)
        train_list, val_list = train_test_split(train_list, test_size=val_size)
        return train_list, val_list, test_list

    def load_images(self):
        self.train_files = glob.glob(os.path.join(self.train_dir, '*.jpg'))
