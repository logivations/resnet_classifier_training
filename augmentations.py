import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensor

def get_augmentations(p=0.5, image_size=(128, 128)):
    imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    train_tfms = A.Compose(
        [
            # A.Crop(x_min=180, y_min=180, x_max=224, y_max=224),
            # A.RandomCrop(image_size, image_size),
            # #A.ShiftScaleRotate(rotate_limit=15, p=p, border_mode=0),
            # A.augmentations.Rotate(limit=15, border_mode=0, p=1.),
            # A.HorizontalFlip(p=0.5),


            A.Resize(*image_size),
            A.RandomResizedCrop(*image_size, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=180, p=0.5),
            # A.Cutout(p=0.5),
            #A.RandomRotate90(p=p),
            A.Flip(p=p),
            # A.OneOf(
            #     [
            #         A.RandomBrightnessContrast(
            #             brightness_limit=0.2,
            #             contrast_limit=0.2,
            #         ),
            #         A.HueSaturationValue(
            #             hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
            #         ),
            #     ],
            #     p=p,
            # ),
            A.OneOf(
                [
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ],
                p=p,
            ),
            A.CoarseDropout(max_holes=10, p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                    # A.IAAPerspective(p=0.3)
                ],
                p=p,
            ),
            ToTensor(normalize=imagenet_stats),
        ]
    )

    valid_tfms = A.Compose(
        [A.Resize(*image_size), ToTensor(normalize=imagenet_stats)]
    )

    return lambda x: train_tfms(image=np.array(x))["image"], lambda x: valid_tfms(image=np.array(x))["image"]