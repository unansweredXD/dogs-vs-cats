import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def preprocess(
        image: np.ndarray,
        transform: A.Compose,
        draw: bool = True
) -> np.ndarray:
    augmented = transform(image=image)["image"]
    if draw:
        visualize(augmented, image)
    return augmented


def load_sample(filename: Path) -> np.ndarray:
    ext = filename.suffix
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(f"Unsupported file extension: {ext}")
    return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)


def visualize(
        image: np.ndarray,
        original_image: np.ndarray | None = None
):
    if original_image is None:
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(image)
    else:
        f, ax = plt.subplots(2, figsize=(10, 8))

        ax[0].imshow(original_image)
        ax[0].set_title("Original image")

        ax[1].imshow(image)
        ax[1].set_title("Augmented image")

    plt.show()
