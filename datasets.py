import albumentations as A
import torch
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path

from image_utils import load_sample, preprocess


class ImageDirsDataset(Dataset):
    def __init__(
            self,
            images_dir: Path,
            preprocessing_config: Optional[Path] = None,
    ):

        self.images_dir = images_dir
        self.ids = [filename.stem for filename in images_dir.iterdir()
                    if filename.is_file() and not filename.stem.startswith('.')]

        if not self.ids:
            raise RuntimeError(f"No input file found in {images_dir}")

        if preprocessing_config:
            self.preprocessing = A.load(str(preprocessing_config), data_format='yaml')
        else:
            self.preprocessing = None

    @staticmethod
    def load_label(filename: str):
        return 0 if filename[:3] == "dog" else 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        image = load_sample(Path(img_file[0]))
        label = self.load_label(name)

        if self.preprocessing:
            image = preprocess(image, transform=self.preprocessing, draw_mode=1)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image.copy()).float().contiguous()

        return {
            "image": image,
            "label": label
        }
