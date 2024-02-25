from pathlib import Path
from datasets import ImageDirsDataset
from torch.utils.data import DataLoader
from config import get_config

config = get_config(Path("conf/base_config.yaml"))

train_dataset = ImageDirsDataset(images_dir=config.dataset.train_path,
                                 preprocessing_config=config.dataset.train_preprocessing_config)

val_dataset = ImageDirsDataset(images_dir=config.dataset.val_path,
                               preprocessing_config=config.dataset.val_preprocessing_config)
test_dataset = ImageDirsDataset(images_dir=config.dataset.test_path,
                                preprocessing_config=config.dataset.val_preprocessing_config)
loader_args = dict(
    batch_size=config.dataset.batch_size,
    num_workers=config.dataset.loader_num_workers,
    pin_memory=True
)

train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
val_loader = DataLoader(val_dataset, shuffle=True, **loader_args)
test_loader = DataLoader(test_dataset, **loader_args)
