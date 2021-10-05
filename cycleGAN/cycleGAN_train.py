from cycleGAN_model import generator, discriminator, down_sampler
from cycleGAN_dataset import cycleGAN_dataset, ImgBuffer
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


class cycleGANTraining:
    """ Class for training cycleGAN generators and discriminators.
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
                            default=1,
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
        parser.add_argument('--n-layers',
                            help='Number of residual layers in generator',
                            default=6,
                            type=int,
                            )
        parser.add_argument('--buffer-size',
                            help='Number of images to store in buffer',
                            default=50,
                            type=int,
                            )
        parser.add_argument('--cycle-ratio',
                            help='The parameter lambda multiplying the cycle loss',
                            default=10.0,
                            type=float,
                            )
        parser.add_argument('--pix-loss',
                            help='use pixel loss',
                            action='store_true',
                            default=False,
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

        # generates data in distribution X from data in distribution Y
        self.generatorX = generator(n_layers=self.cli_args.n_layers).to(self.device)
        # generates data in distribution Y from data in distribution X
        self.generatorY = generator(n_layers=self.cli_args.n_layers).to(self.device)
        # discriminator for data in distribution X
        self.discriminatorX = discriminator().to(self.device)
        # discriminator for data in distribution Y
        self.discriminatorY = discriminator().to(self.device)
        # down_sampler: to down-sample, increase pool_size.
        # Here, we also provide option to blur before down-sampling
        # Default: identity loss as defined in cycle GAN paper
        if self.cli_args.pix_loss:
            self.down_sampler = down_sampler(blur=False, pool_size=1).to(self.device)

        self.optimizer_G = torch.optim.Adam(
            list(self.generatorX.parameters()) +
            list(self.generatorY.parameters()),
            lr=0.0002, betas=(0.5, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.discriminatorX.parameters()) +
            list(self.discriminatorY.parameters()),
            lr=0.0002, betas=(0.5, 0.999),
        )

        self.lossfn_GAN = nn.MSELoss()
        self.lossfn_cycle = nn.L1Loss()
        self.lossfn_pix = nn.L1Loss()

    def main(self):
        """ Main function for training generator and discriminator.
        We save generated images every 10 epochs, and at the end of the training.
        If the argument --save-model is given, saves model every 10 epochs.
        """
        self.trn_dlX, self.trn_dlY = self.initTrainDl()
        self.val_dlX, self.val_dlY = self.initValDl()
        self.ImgBufferX = ImgBuffer(self.cli_args.buffer_size)
        self.ImgBufferY = ImgBuffer(self.cli_args.buffer_size)
        self.lr_scheduler_list = self.init_lr_scheduler()

        for epoch in tqdm(range(1, self.cli_args.epochs + 1), desc='full training loop'):
            self.doTraining()
            for lr_scheduler in self.lr_scheduler_list:
                lr_scheduler.step()
            if epoch % 10 == 1 or epoch == self.cli_args.epochs:
                self.visualize('val', epoch)
                if self.cli_args.save_model:
                    self.saveModel(epoch)

    def initTrainDl(self):
        """ Initialize the training dataloader for two datasets x and y
        When we loop over the datasets, we zip them.
        This means that if the lengths of the datasets differ,
        the shortest length is taken to be the effective length of the dataset.
        It should not cause any problem if we put shuffle=True.
        """
        pin_memory = True if self.use_cuda else False
        datasetX = torch.utils.data.DataLoader(cycleGAN_dataset(self.cli_args.data_folder,
                                                                train_bool=True,
                                                                x_or_y=0,
                                                                jitter=self.cli_args.jitter,
                                                                flip=self.cli_args.flip,
                                                                ),
                                               batch_size=self.cli_args.batch_size,
                                               shuffle=True,
                                               num_workers=self.cli_args.num_workers,
                                               pin_memory=pin_memory
                                               )
        datasetY = torch.utils.data.DataLoader(cycleGAN_dataset(self.cli_args.data_folder,
                                                                train_bool=True,
                                                                x_or_y=1,
                                                                jitter=self.cli_args.jitter,
                                                                flip=self.cli_args.flip,
                                                                ),
                                               batch_size=self.cli_args.batch_size,
                                               shuffle=True,
                                               num_workers=self.cli_args.num_workers,
                                               pin_memory=pin_memory
                                               )
        return datasetX, datasetY

    def initValDl(self):
        """ Initialize the validation dataloader for two datasets x and y.
        Note that data is shuffled.
        """
        pin_memory = True if self.use_cuda else False
        datasetX = torch.utils.data.DataLoader(cycleGAN_dataset(self.cli_args.data_folder,
                                                                train_bool=False,
                                                                x_or_y=0,
                                                                jitter=self.cli_args.jitter,
                                                                flip=self.cli_args.flip,
                                                                ),
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=self.cli_args.num_workers,
                                               pin_memory=pin_memory
                                               )
        datasetY = torch.utils.data.DataLoader(cycleGAN_dataset(self.cli_args.data_folder,
                                                                train_bool=False,
                                                                x_or_y=1,
                                                                jitter=self.cli_args.jitter,
                                                                flip=self.cli_args.flip,
                                                                ),
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=self.cli_args.num_workers,
                                               pin_memory=pin_memory
                                               )
        return datasetX, datasetY

    def init_lr_scheduler(self):
        """Setup scheduler for learning rate. 
        Following the cycleGAN paper, we train for first half
        of the total training epochs before linearly decaying
        the learing rate to zero.
        """
        def lr_lambda(epoch):
            lr_factor = 1.0-max(0, epoch-self.cli_args.epochs/2.0)/(self.cli_args.epochs/2.0+1.0)
            return lr_factor
        return [torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lr_lambda), torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)]

    def doTraining(self):
        """ Train for an epoch by looping over the training dataset.
        Note that we zip trn_dlX and trn_dlY, so that if the two datasets have different lengths,
        we loop over the shorter length.
        """
        for data_X, data_Y in zip(self.trn_dlX, self.trn_dlY):
            self.discriminatorX.set_requires_grad(False)
            self.discriminatorY.set_requires_grad(False)
            data_X = data_X.to(self.device)
            data_Y = data_Y.to(self.device)
            fake_X = self.generatorX(data_Y)
            fake_Y = self.generatorY(data_X)
            loss_G = self.compute_loss_G(data_X, data_Y, fake_X, fake_Y)
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            self.discriminatorX.set_requires_grad(True)
            self.discriminatorY.set_requires_grad(True)
            fake_X = self.ImgBufferX(fake_X.detach())
            fake_Y = self.ImgBufferY(fake_Y.detach())
            loss_D = self.compute_loss_D(data_X, data_Y, fake_X, fake_Y)
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

    def compute_loss_G(self, data_X, data_Y, fake_X, fake_Y):
        """ Compute the loss for generator. Here, we use LSGAN loss.
        Args:
            data_X (torch.tensor): data in the distribution X
            data_Y (torch.tensor): data in the distribution Y
            fake_X (torch.tensor): fake data produced by generatorX
            fake_Y (torch.tensor): fake data produced by generatorY

        Returns:
            torch.tensor: the loss for generator
        """
        recon_X = self.generatorX(fake_Y)
        recon_Y = self.generatorY(fake_X)

        patch_fake_X = self.discriminatorX(fake_X)
        patch_fake_Y = self.discriminatorY(fake_Y)

        loss_GAN = self.lossfn_GAN(patch_fake_X, torch.ones_like(patch_fake_X)) \
            + self.lossfn_GAN(patch_fake_Y, torch.ones_like(patch_fake_Y))
        loss_cycle = self.lossfn_cycle(recon_X, data_X) \
            + self.lossfn_cycle(recon_Y, data_Y)

        loss_G = loss_GAN + self.cli_args.cycle_ratio * loss_cycle
        if self.cli_args.pix_loss:
            loss_pix = self.lossfn_pix(self.down_sampler(fake_Y), self.down_sampler(data_X)) \
                + self.lossfn_pix(self.down_sampler(fake_X),self.down_sampler(data_Y))
            loss_G += 5.0*loss_pix

        return loss_G

    def compute_loss_D(self, data_X, data_Y, fake_X, fake_Y):
        """ Compute the loss for discriminator.
        Args:
            data_X (torch.tensor): data in the distribution X
            data_Y (torch.tensor): data in the distribution Y
            fake_X (torch.tensor): fake data produced by generatorX
            fake_Y (torch.tensor): fake data produced by generatorY

        Returns:
            torch.tensor: the loss for discriminator
        """
        patch_real_X = self.discriminatorX(data_X)
        patch_real_Y = self.discriminatorY(data_Y)
        patch_fake_X = self.discriminatorX(fake_X)
        patch_fake_Y = self.discriminatorY(fake_Y)

        loss_real = self.lossfn_GAN(patch_real_X, torch.ones_like(patch_real_X)) \
            + self.lossfn_GAN(patch_real_Y, torch.ones_like(patch_real_Y))
        loss_fake = self.lossfn_GAN(patch_fake_X, torch.zeros_like(patch_fake_X)) \
            + self.lossfn_GAN(patch_fake_Y, torch.zeros_like(patch_fake_Y))

        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_D

    def visualize(self, mode, epoch):
        """ Save generated image.
        Args:
            mode (str): 'trn' or 'val'. The two options correspond to sampling and generating image from training/validation dataset.
            epoch (int): epoch at which to generate and save image. Note that the file name contains the epoch number.
        """
        if mode == 'trn':
            dlX = self.trn_dlX
            dlY = self.trn_dlY
            file_path = os.path.join('visualize', self.cli_args.data_folder, 'trn')
        elif mode == 'val':
            dlX = self.val_dlX
            dlY = self.val_dlY
            file_path = os.path.join('visualize', self.cli_args.data_folder, 'val')
        else:
            file_path = os.path.join('visualize', self.cli_args.data_folder, 'trn')
            logger.warning('warning: unrecognized option' + str(mode))
            logger.warning('using mode=trn')
            dlX = self.trn_dlX
            dlY = self.trn_dlY

        os.makedirs(file_path, mode=0o755, exist_ok=True)

        for data_X, data_Y in zip(dlX, dlY):
            data_X = data_X.to(self.device)
            data_Y = data_Y.to(self.device)

            with torch.no_grad():
                img_batch_G_X = self.generatorX(data_Y)
                img_batch_G_Y = self.generatorY(data_X)
                data_X = (data_X+1.0)/2
                data_Y = (data_Y+1.0)/2
                img_batch_G_X = (img_batch_G_X+1.0)/2
                img_batch_G_Y = (img_batch_G_Y+1.0)/2
                torchvision.utils.save_image(data_X, os.path.join(
                    file_path, 'x' + str(epoch) + '.png'))
                torchvision.utils.save_image(data_Y, os.path.join(
                    file_path, 'y' + str(epoch) + '.png'))
                torchvision.utils.save_image(img_batch_G_X, os.path.join(
                    file_path, 'x_G' + str(epoch) + '.png'))
                torchvision.utils.save_image(img_batch_G_Y, os.path.join(
                    file_path, 'y_G' + str(epoch) + '.png'))
            break

    def saveModel(self, epoch):
        """ Saves model to a subdirectory of 'model' having the dataset name.
        Saved values: sys.argv, generator.state_dict, discriminator.state_dict, optimizer_G.state_dict, optimizer_D.state_dict (for both X and Y).
        """
        file_path = os.path.join('model', self.cli_args.data_folder)
        os.makedirs(file_path, mode=0o755, exist_ok=True)
        state = {
            'sys_argv': sys.argv,
            'epoch': epoch,
            'generatorX_state': self.generatorX.state_dict(),
            'generatorY_state': self.generatorY.state_dict(),
            'discriminatorX_state': self.discriminatorX.state_dict(),
            'discriminatorY_state': self.discriminatorY.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
        }
        torch.save(state, os.path.join(file_path, 'cycleGAN.state'))
        # logger.info('saving model')


if __name__ == '__main__':
    x = cycleGANTraining()
    x.main()
