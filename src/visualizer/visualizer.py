import os

import numpy as np
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import torch.nn.functional as F
import pandas as pd
import albumentations as A
import cv2
import copy
import torch


class Visualizer:
    @staticmethod
    def display_image_grid(images_filepaths, predicted_labels=None, rows=2, cols=5, name=''):
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 6))
        random_idx = np.random.randint(1, len(images_filepaths), size=10)
        i = 0
        for idx in random_idx:
            image = cv2.imread(images_filepaths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            true_label = images_filepaths[idx].split(os.sep)[-1].split('_')[0]
            if predicted_labels is not None:
                class_ = predicted_labels[idx]
                color = 'green' if true_label == class_ else 'red'
            else:
                class_ = true_label
                color = 'green'
            ax.ravel()[i].imshow(image.astype(np.uint8))
            ax.ravel()[i].set_title(class_, color=color)
            ax.ravel()[i].set_axis_off()
            i += 1
        figure.suptitle(name, fontsize=24)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 6))

        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        figure.suptitle('Пример аугментации', fontsize=24)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_predict(model, test_loader, device):
        model = model.eval()
        predicted_labels = []
        with torch.no_grad():
            for data, fileid in test_loader:
                data = data.to(device, non_blocking=True)
                output = model(data)
                predictions = F.softmax(output, dim=1)[:, 1].tolist()
                predicted_labels += list(zip(list(fileid), predictions))

        predicted_labels.sort(key=lambda x: int(x[0]))
        idx = list(map(lambda x: x[0], predicted_labels))
        prob = list(map(lambda x: x[1], predicted_labels))
        submission = pd.DataFrame({'id': idx, 'label': prob})

        preds = pd.DataFrame(columns=['id', 'class'])

        for i in range(len(submission)):
            label = submission.label[i]
            if label > 0.5:
                label = 'dog'
            else:
                label = 'cat'
            preds.loc[len(preds.index)] = [submission.id[i], label]
