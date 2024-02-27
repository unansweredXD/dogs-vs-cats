import os
import random

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn


class Dataset(nn.Module):
    def __init__(self, file_list, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_list = file_list
        self.transform = transform
        self.images = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        img = cv2.imread(img_path)

        while img is None:
            label = img_path.split(os.sep)[-1].split('_')[0]

            img = cv2.imread('dataset/data/' + label + '_' + str(random.randint(0, 10000)) + '.jpg')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = img_path.split(os.sep)[-1].split('_')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0

        img = self.transform(image=img)['image']

        return img, label

    def display_image_grid(self, predicted_labels=pd.DataFrame(), rows=2, cols=5):
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 12))
        random_idx = np.random.randint(1, len(self.file_list), size=10)
        i = 0

        for idx in random_idx:
            image = cv2.imread(self.file_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            correct_label = self.file_list[idx].split(os.sep)[-1].split('_')[0]
            if predicted_labels.empty:
                image_class = correct_label
                color = 'green'
            else:
                image_class = predicted_labels.loc[predicted_labels['id'] == correct_label, 'class'].values[0]
                color = 'red'
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_title(image_class, color=color)
            ax.ravel()[i].set_axis_off()
            i += 1

        plt.tight_layout()
        plt.show()
