import albumentations as A
from albumentations.pytorch import ToTensorV2


transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), always_apply=False, p=0.5),
        A.RandomCrop(height=148, width=148),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.452, 0.406), std=(0.229, 0.228, 0.225)),
        ToTensorV2(),
    ])

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=148, width=148),
        A.Normalize(mean=(0.485, 0.452, 0.406), std=(0.229, 0.228, 0.225)),
        ToTensorV2(),
    ])
