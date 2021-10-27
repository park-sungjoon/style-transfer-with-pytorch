import torch
from torch.utils.data import Dataset
import os
import functools
import glob
import random
from PIL import Image
import torchvision.transforms as transforms


@functools.lru_cache(4)
def getImgPaths(dataset_folder, train_bool):
    if train_bool:
        trn_or_test = 'train'
    else:
        trn_or_test = 'test'
    dataset_path = os.path.join('../datasets', dataset_folder, trn_or_test)
    img_paths_x = glob.glob(os.path.join(dataset_path, 'x', '*.*'))
    img_paths_y = glob.glob(os.path.join(dataset_path, 'y', '*.*'))
    img_paths_x = list(path for path in img_paths_x if os.path.splitext(path)[
                       1] in ['.jpg', '.png'])
    img_paths_y = list(path for path in img_paths_y if os.path.splitext(path)[
                       1] in ['.jpg', '.png'])
    return img_paths_x, img_paths_y


class CUT_dataset(Dataset):
    """ Dataset class for CUT.

    We assume that the folder containing the data set is located in the folder ../datasets/
    The images should be organized in the form ../datasets/train, ../datasets/test
    Within the train and test folders, there should be two folders containing the images
    distributed according to x and y (we transfer styles from x to y and from y to x).
    Note that these folders are named 'x' and 'y' (i.e. '../datasets/train/x', '../datasets/train/y', etc)
    """

    def __init__(self, dataset_folder, train_bool, x_or_y, normalize=True, jitter=True, flip=True):
        """ Initialize dataset.
        Args:
            dataset_folder (str): name of folder containing the data. E.g. 'facades'
            train_bool (bool): if True, use '/datasets/facades/train', else, use '/datasets/facades/test' (for facades dataset)
            x_or_y (int): 0 for x, 1 for y
            noramlize (bool): normalize image so that pixels lie between -1 and 1.
            jitter (bool): if True, enlarge image and randomly crop image (data augmentation).
            flip (bool): if True, randomly flip image horizontally (data augmentation).
        """
        self.img_paths = getImgPaths(dataset_folder, train_bool)[x_or_y]
        self.len = len(self.img_paths)
        self.jitter = jitter
        self.flip = flip
        self.normalize = normalize

    def __len__(self):
        return self.len

    def __getitem__(self, ndx):
        # PIL image of shape channels x columns x rows
        img = Image.open(self.img_paths[ndx]).convert('RGB')
        img_t = transform(img, normalize=self.normalize,
                          jitter=self.jitter, flip=self.flip)
        return img_t


def transform(img, normalize=True, jitter=True, flip=True):
    """ Transform image (see pix2pix paper).
    Args:
        normalize (bool): normalize image to lie between [-1,1]
        jitter (bool): resize 256 by 256 image to 286 by 286 and randomly crop image.
        flip (bool): randomly flip image horizontally.
    """
    transform_list = []
    if jitter:
        transform_list.append(transforms.Resize(
            286, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(256))
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)(img)


class ImgBuffer():
    """Class that keeps buffer of images"""

    def __init__(self, buffer_size):
        """
        Args:
            buffer_size (int): number of images to store
        """
        self.buffer_size = buffer_size
        self.buffer_list = []
        self.buffer_count = 0

    def __call__(self, img_t):
        """ Takes batch of images, stores into buffer, 
        returns batch of images of the same size from the buffer.

        Args:
            img_t (torch.tensor): batch of images to put into buffer

        Returns:
            torch.tensor: batch of images from buffer.
        """
        if self.buffer_size == 0:
            return img_t
        return_list = []
        for i in range(img_t.shape[0]):
            if self.buffer_count < self.buffer_size:
                self.buffer_count += 1
                self.buffer_list.append(img_t[i, :].unsqueeze(0))
                return_list.append(img_t[i, :].unsqueeze(0))
            else:
                return_bool = random.randint(0, 1)
                if return_bool:
                    return_list.append(img_t[i].unsqueeze(0))
                else:
                    pop_int = random.randint(0, self.buffer_size - 1)
                    return_list.append(self.buffer_list.pop(pop_int))
                    self.buffer_list.append(img_t[i, :].unsqueeze(0))
        return torch.cat(return_list, dim=0)
