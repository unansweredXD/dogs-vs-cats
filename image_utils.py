import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional
from pathlib import Path


def preprocess(
        image: np.ndarray,
        transform: A.Compose,
        # 0 - False, 1 - draw augment too, 2 - draw for predict
        draw_mode: int = 0
) -> np.ndarray:
    augmented = transform(image=image)["image"]
    if draw_mode == 1:
        visualize(augmented, draw_mode, image)
    elif draw_mode == 2:
        visualize(augmented, draw_mode)
    return augmented


def load_sample(filename: Path) -> np.ndarray:
    ext = filename.suffix
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(f"Unsupported file extension: {ext}")
    return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)


def visualize(
        image: np.ndarray = None,
        mode: int = 1,
        original_image: Optional[np.ndarray] = None

):
    if mode == 2:
        plt.imshow(image)

    elif mode == 1:
        f, ax = plt.subplots(2, figsize=(10, 8))

        ax[0].imshow(original_image)
        ax[0].set_title("Original image")

        ax[1].imshow(image)
        ax[1].set_title("Augmented image")

    plt.show()
