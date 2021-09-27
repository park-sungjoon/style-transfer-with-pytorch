from pix2pix_model import generator, discriminator
from pix2pix_dataset import pix2pix_dataset
import argparse
import sys
import torch
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


class pix2pixTraining:
    """ Class for training pix2pix generator and discriminator.
    """

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--data-folder',
                            help='Name of data folder',
                            default='facades',
                            type=str,
                            )

        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=8,
                            type=int,
                            )

        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker process for background data loading',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--flip',
                            help='Disable the default behavior of augmenting the training data by randomly flipping the images horizontally',
                            action='store_false',
                            default=True,
                            )

        parser.add_argument('--jitter',
                            help='Disable the default behavior of augmenting the training data by enlarging enlarging image and randomly cropping it',
                            action='store_false',
                            default=True,
                            )

        parser.add_argument('--save-model',
                            help='Save the model',
                            action='store_true',
                            default=False,
                            )

        self.cli_args = parser.parse_args(sys_argv)

        logger.info('received following arguments:\n' + str(vars(self.cli_args)))

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f'Using device {self.device}')

        self.generator = generator().to(self.device)
        self.discriminator = discriminator().to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func_L1 = torch.nn.L1Loss()

    def main(self):
        """ Main function for training generator and discriminator. 
        We save generated images every 10 epochs, and at the end of the training.
        If the argument --save-model is given, saves model.
        """
        self.trn_dl = self.initTrainDl()
        self.val_dl = self.initValDl()
        for epoch in tqdm(range(1, self.cli_args.epochs + 1), desc='full training loop'):
            self.doTraining()
            if epoch % 10 == 1 or epoch == self.cli_args.epochs:
                self.visualize('val', epoch)
        if self.cli_args.save_model:
            self.saveModel()

    def initTrainDl(self):
        """ Initialize the training dataloader"""
        pin_memory = True if self.use_cuda else False
        dataset = torch.utils.data.DataLoader(pix2pix_dataset(self.cli_args.data_folder,
                                                              train_bool=True,
                                                              jitter=self.cli_args.jitter,
                                                              flip=self.cli_args.flip,
                                                              ),
                                              batch_size=self.cli_args.batch_size,
                                              shuffle=True,
                                              num_workers=self.cli_args.num_workers,
                                              pin_memory=pin_memory
                                              )
        return dataset

    def initValDl(self):
        """ Initialize the validation dataloader"""
        pin_memory = True if self.use_cuda else False
        dataset = torch.utils.data.DataLoader(pix2pix_dataset(self.cli_args.data_folder,
                                                              train_bool=False,
                                                              jitter=self.cli_args.jitter,
                                                              flip=self.cli_args.flip,
                                                              ),
                                              batch_size=8,
                                              shuffle=False,
                                              num_workers=self.cli_args.num_workers,
                                              pin_memory=pin_memory
                                              )
        return dataset

    def doTraining(self):
        """ Train for an epoch by looping over the training dataset """
        for data_x, data_y in tqdm(self.trn_dl, desc='current loop' + ' ' * 6, leave=False):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

            self.optimizer_D.zero_grad()
            self.discriminator.set_requires_grad(True)
            loss_D, fake_data = self.compute_loss_D(data_x, data_y)
            loss_D.backward()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            self.discriminator.set_requires_grad(False)
            loss_G = self.compute_loss_G(data_x, data_y, fake_data)
            loss_G.backward()
            self.optimizer_G.step()

    def compute_loss_D(self, data_x, data_y):
        """ Compute the loss for discriminator.
        Args:
            data_x (torch.tensor): data in the distribution x (the data sampled by the generator)
            data_y (torch.tensor): data in the distribution y (the target data for generator)

        Returns:
            torch.tensor: the loss for discriminator
        """
        patch_result_real = self.discriminator(torch.cat((data_x, data_y), dim=1))
        loss_real = self.loss_func(patch_result_real, torch.ones_like(patch_result_real, requires_grad=False))

        fake_data = self.generator(data_x)
        patch_result_fake = self.discriminator(torch.cat((data_x, fake_data.detach()), dim=1))
        loss_fake = self.loss_func(patch_result_fake, torch.zeros_like(patch_result_fake, requires_grad=False))

        # In the paper, the loss is loss_real+loss_fake but in the code, these two are averaged.
        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_D, fake_data

    def compute_loss_G(self, data_x, data_y, fake_data, l1_lambda=100.0):
        """ Compute the loss for generator.
        Args:
            data_x (torch.tensor): data in the distribution x (the data sampled by the generator)
            data_y (torch.tensor): data in the distribution y (the target data for generator)
            fake_data (torch.tensor): the fake data produced by generator (that is supposed to mimick data_y)
            l1_lambda: hyperparameter in the loss function

        Returns:
            torch.tensor: the loss for generator
        """
        patch_result_fake = self.discriminator(torch.cat((data_x, fake_data), dim=1))
        loss_GAN = self.loss_func(patch_result_fake, torch.ones_like(patch_result_fake, requires_grad=False))
        loss_L1 = self.loss_func_L1(data_y, fake_data) * l1_lambda

        loss_G = loss_GAN + loss_L1

        return loss_G

    def visualize(self, mode, epoch):
        """ Save generated image.
        Args:
            mode (str): 'trn' or 'val'. The two options correspond to sampling and generating image from training/validation dataset.
            epoch (int): epoch at which to generate and save image. Note that the file name contains the epoch number.
        """
        if mode == 'trn':
            dl = self.trn_dl
            file_path = 'visualize/trn'
        elif mode == 'val':
            dl = self.val_dl
            file_path = 'visualize/val'
        else:
            file_path = 'visualize/trn'
            logger.warning('warning: unrecognized option' + str(mode))
            logger.warning('using mode=trn')
            dl = self.trn_dl

        os.makedirs(file_path, mode=0o755, exist_ok=True)

        for data_x, data_y in dl:
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

            with torch.no_grad():
                img_batch_G = self.generator(data_x)
                data_x = (data_x+1.0)/2.0
                data_y = (data_y+1.0)/2.0
                img_batch_G = (img_batch_G+1.0)/2.0
                torchvision.utils.save_image(data_x, os.path.join(file_path, 'x' + str(epoch) + '.png'))
                torchvision.utils.save_image(img_batch_G, os.path.join(file_path, 'y_G' + str(epoch) + '.png'))
                torchvision.utils.save_image(data_y, os.path.join(file_path, 'y' + str(epoch) + '.png'))
            break

    def saveModel(self):
        """ Saves model to folder '/model'
        Saved values: sys.argv, generator.state_dict, discriminator.state_dict, optimizer_G.state_dict, optimizer_D.state_dict.
        """
        file_path = 'model/'
        os.makedirs(file_path, mode=0o755, exist_ok=True)
        state = {
            'sys_argv': sys.argv,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
        }
        torch.save(state, os.path.join(file_path, 'pix2pix.state'))
        logger.info('saving model')


if __name__ == '__main__':
    x = pix2pixTraining()
    x.main()
