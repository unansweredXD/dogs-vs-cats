import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.autonotebook import tqdm
import seaborn as sns
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt


def test_model(model, test_loader, device):
    model = model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    with torch.inference_mode():
        for data in tqdm(test_loader):
            images, labels = data['image'].to(device), data['label'].to(device)

            outputs = model(images)
            predictions, _ = torch.max(outputs, 1)
            predictions = torch.round(predictions)

            y_true = np.concatenate((y_true, labels.cpu().numpy()))
            y_pred = np.concatenate((y_pred, predictions.cpu().numpy()))

    conf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in ['dog', 'cat']],
                         columns=[i for i in ['dog', 'cat']])
    sns.heatmap(df_cm, annot=True, center=0, cmap='coolwarm', fmt='g', cbar=True)
    plt.show()

    f1 = f1_score(y_true, y_pred, average='binary')
    print(f'f1 on test data: {f1:.4f}')

    return f1
