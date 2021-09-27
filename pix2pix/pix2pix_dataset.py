import torch
from torch.utils.data import Dataset
import os
import functools
import glob
import random
from PIL import Image
import torchvision.transforms as transforms


@functools.lru_cache(2)
def getImgPaths(dataset_folder, train_bool):
    if train_bool:
        trn_or_test = 'train'
    else:
        trn_or_test = 'test'
    dataset_path = os.path.join('../datasets', dataset_folder, trn_or_test)
    img_paths = glob.glob(os.path.join(dataset_path, '*.*'))
    img_paths = list(path for path in img_paths if os.path.splitext(path)[1] in ['.jpg', '.png'])
    return img_paths


class pix2pix_dataset(Dataset):
    """ Dataset class for pix2pix.

    We assume that the folder containing the data set is located in the folder ../datasets/
    """

    def __init__(self, dataset_folder, train_bool, reverse_xy=True, normalize=True, jitter=True, flip=True):
        """ Initialize dataset.
        Args:
            dataset_folder (str): name of folder containing the data. E.g. 'facades'
            train_bool (bool): if True, use '/datasets/facades/train', else, use '/datasets/facades/test'
            reverse_xy (bool): swap x and y labels. Note that if reverse_xy=False, 
                x is the  left half of image in '/datasets/facades/...', and
                y is the right half of the image in '/datasets/facades/...'.
                Also, note that x is input to generator, and y is the target for generator.
            noramlize (bool): normalize image so that pixels lie between -1 and 1.
            jitter (bool): if True, enlarge image and randomly crop image (data augmentation).
            flip (bool): if True, randomly flip image horizontally (data augmentation).
        """
        self.img_paths = getImgPaths(dataset_folder, train_bool)
        self.len = len(self.img_paths)
        print(Image.open(self.img_paths[0]).size)
        self.width, self.height = Image.open(self.img_paths[0]).size
        self.half_width = self.width // 2
        self.jitter = jitter
        self.flip = flip
        self.normalize = normalize
        self.reverse_xy = reverse_xy

    def __len__(self):
        return self.len

    def __getitem__(self, ndx):
        img = Image.open(self.img_paths[ndx])  # PIL image of shape channels x columns x rows
        x_img = img.crop((0, 0, self.half_width, self.height))
        y_img = img.crop((self.half_width, 0, self.width, self.height))
        x_t = transform(x_img, normalize=self.normalize, jitter=self.jitter, flip=self.flip)
        y_t = transform(y_img, normalize=self.normalize, jitter=self.jitter, flip=self.flip)
        if self.reverse_xy:
            return y_t, x_t
        else:
            return x_t, y_t


def transform(img, normalize=True, jitter=True, flip=True):
    """ Transform image as required in pix2pix_dataset
    """
    transform_list = []
    if jitter:
        transform_list.append(transforms.Resize(286,  transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(256))
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)(img)
