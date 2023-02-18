import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from .custom_dataset import ImageFolderRemapPair


class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args):

    mean = [0.5]
    std = [0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                         transforms.ToTensor(),
                         normalize])

    img_dir = args.data_dir

    dataset = ImageFolderRemapPair(
        img_dir,
        transform=transform,
        input_ch=1,
        pair_size=args.pair_size,
    )

    return dataset
