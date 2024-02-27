import os
import random
import numpy as np
import torch
import wandb
from torch import optim, nn
from torch.utils.data import DataLoader
from src.dataset.augmentation import transform, val_transforms
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.metrics.wandb import init_wandb
from src.model.model import CNN
from src.model.train import ModelTrainer
from src.visualizer.visualizer import Visualizer

EPOCH = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = 52


def main():
    wandb.login()
    init_wandb(
        learning_rate=LEARNING_RATE,
        epochs=EPOCH,
        batch_size=BATCH_SIZE
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'device: {device}')

    loader = DatasetLoader()
    loader.extract_dataset()

    if not os.path.isdir('dataset/data'):
        loader.split_dataset()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_list, val_list, test_list = loader.split_train(test_size=0.2, val_size=0.4)
    Visualizer().display_image_grid(train_list, name='Пример данных в датасете')

    train_data = Dataset(train_list, transform=transform)
    test_data = Dataset(test_list, transform=transform)
    val_data = Dataset(val_list, transform=val_transforms)

    Visualizer().visualize_augmentations(train_data)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=EPOCH
    )
    trainer.start_training()
    torch.save(model.state_dict(), 'model.pt')

    Visualizer().display_predict(
        model,
        test_loader,
        device
    )


if __name__ == '__main__':
    main()
