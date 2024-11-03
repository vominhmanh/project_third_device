# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT
# from src.data_io import transform as trans
import torchvision.transforms as trans
import imgaug.augmenters as iaa
import imgaug.parameters as iap

def get_valid_augmentation():
    seq_valid = iaa.Sequential([
        iaa.Sometimes(0.15, iaa.Crop(px=(1,4), keep_size=False)),
        iaa.Sometimes(0.15, 
            iaa.Affine(
                rotate=(-2, 2),
                order=[0, 1],
                cval=(0,255),
                mode='edge',
            )
        ),
        
        # Low resolution, compressed image
        iaa.Sometimes(0.05,
            iaa.OneOf([
                iaa.imgcorruptlike.Pixelate(severity=1),
                iaa.imgcorruptlike.JpegCompression(severity=1),
                iaa.UniformColorQuantization(n_colors=(30, 256)),
            ])
        ),
        
        # Low light condition
        iaa.Sometimes(0.25, 
            iaa.Sequential([
                iaa.OneOf([
                    iaa.AdditivePoissonNoise((1, 10), per_channel=True),
                    iaa.AdditivePoissonNoise((1, 10)),
                    iaa.AdditiveLaplaceNoise(scale=(0.005 * 255, 0.05 * 255)),
                    iaa.AdditiveLaplaceNoise(scale=(0.005 * 255, 0.05 * 255), per_channel=True)
                ]),
                iaa.JpegCompression(compression=(5, 80)),
            ])
        ),

        # Normal blur
        iaa.Sometimes(0.2,
            iaa.OneOf([
                iaa.GaussianBlur(1),
                iaa.AverageBlur(k=(2, 3)),
                iaa.MotionBlur((3, 4)),
            ]),
        ),
        
        # Temperature 
        iaa.Sometimes(0.05, 
            iaa.OneOf([
                iaa.ChangeColorTemperature((5000, 12000)),
                iaa.AddToHue((-5,5)),
                iaa.AddToSaturation((-5,5))
            ]),
        ),
        # Random brightness and color
        iaa.Sometimes(0.25, iaa.OneOf([
            iaa.Add(iap.Normal(iap.Choice([-20, 20]), 10)),
            iaa.Multiply((0.85, 1.15)),
            iaa.AddToBrightness((-30, 30)),
            iaa.MultiplyBrightness((0.85, 1.15)),
            iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-10, 10)),
            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-20, 20]), 10)), start_at=(0, 0.2), end_at=(0.8, 1)),
            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-20, 20]), 10)), start_at=(0.8, 1), end_at=(0, 0.2)),
        ]))
    ])

    return seq_valid


def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize((80, 80)),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    valid_augment = get_valid_augmentation()
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path, train_transform, valid_augment,
                               None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    return train_loader
