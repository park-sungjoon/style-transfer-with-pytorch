from CUT_model import generator, discriminator, MLP_bundle, NCE_loss
from CUT_dataset import CUT_dataset, ImgBuffer
import argparse
import sys
import torch
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm
import logging
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


class CUTTraining:
    """ Class for training CUT generators and discriminators.
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
                            default=9,
                            type=int,
                            )
        parser.add_argument('--feat-str',
                            help='Which features to use for NCE loss',
                            default='pix, d128, d256, R256_1, R256_5',
                            type=str,
                            )
        parser.add_argument('--buffer-size',
                            help='Number of images to store in buffer. I think CUT uses buffer size of 0',
                            default=50,
                            type=int,
                            )
        parser.add_argument('--num-patches',
                            help='Number of patches for NCE loss',
                            default=256,
                            type=int,
                            )
        parser.add_argument('--NCE-ratio',
                            help='Ratio for NCE loss for source distribution',
                            default=1.0,
                            type=float,
                            )
        parser.add_argument('--NCE-ratio-id',
                            help='Ratio for NCE loss for target distribution (identity loss)',
                            default=1.0,
                            type=float,
                            )
        parser.add_argument('--save-model',
                            help='Save the model',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--swap-xy',
                            help='Switch target distribution. By default, target distribution is Y.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--flip-equivariance',
                            help='Enforce flip equivariance.',
                            action='store_true',
                            default=False,
                            )

        self.cli_args = parser.parse_args(sys_argv)

        logger.info('received following arguments:\n' +
                    str(vars(self.cli_args)))

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f'Using device {self.device}')

        # translates data to the target distribution (default target is y)
        self.generator = generator(
            n_layers=self.cli_args.n_layers,
            feat_str_list=[x.strip()
                           for x in self.cli_args.feat_str.split(',')],
        ).to(self.device)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0002, betas=(0.5, 0.999),
        )
        # discriminator for data in target distribution
        self.discriminator = discriminator().to(self.device)
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002, betas=(0.5, 0.999),
        )
        # MLP layers for contrasting patches.
        # The complete setup is done during the first forward pass with MLP_bundle.setup_MLP
        self.MLP_bundle = MLP_bundle().to(self.device)
        self.MLP_bundle.set_device(self.device)
        self.setup = False

        self.lossfn_GAN = nn.MSELoss()
        self.lossfn_NCE_list = [NCE_loss(batch_size=self.cli_args.batch_size) for layer in [
            x.strip() for x in self.cli_args.feat_str.split(',')]]

    def main(self):
        """ Main function for training generator, discriminator, and MLP bundle.
        We save generated images every 10 epochs, and at the end of the training.
        If the argument --save-model is given, saves model every 10 epochs.
        """
        self.trn_dlX, self.trn_dlY = self.initTrainDl()
        self.val_dlX, self.val_dlY = self.initValDl()
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
        datasetX = torch.utils.data.DataLoader(CUT_dataset(self.cli_args.data_folder,
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
        datasetY = torch.utils.data.DataLoader(CUT_dataset(self.cli_args.data_folder,
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
        if self.cli_args.swap_xy:
            return datasetY, datasetX
        else:
            return datasetX, datasetY

    def initValDl(self):
        """ Initialize the validation dataloader for two datasets x and y.
        Note that data is shuffled.
        """
        pin_memory = True if self.use_cuda else False
        datasetX = torch.utils.data.DataLoader(CUT_dataset(self.cli_args.data_folder,
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
        datasetY = torch.utils.data.DataLoader(CUT_dataset(self.cli_args.data_folder,
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
        if self.cli_args.swap_xy:
            return datasetY, datasetX
        else:
            return datasetX, datasetY

    def init_lr_scheduler(self):
        """Setup scheduler for learning rate. 
        As in the cycleGAN setup, we train for first half of the total training epochs 
        before linearly decaying the learing rate to zero.
        """
        def lr_lambda(epoch):
            lr_factor = 1.0 - \
                max(0, epoch - self.cli_args.epochs / 2.0) / \
                (self.cli_args.epochs / 2.0 + 1.0)
            return lr_factor
        return [torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lr_lambda), torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)]

    def doTraining(self):
        """ Train for an epoch by looping over the training dataset.
        Note that we zip trn_dlX and trn_dlY, so that if the two datasets have different lengths,
        we loop over the shorter length.
        """
        for data_X, data_Y in zip(self.trn_dlX, self.trn_dlY):
            data_X = data_X.to(self.device)
            data_Y = data_Y.to(self.device)
            # Generate fake data. data_Y is processed with generator for identity loss.
            fake_Y_cat, feat_list_cat = self.generator(
                torch.cat([data_X, data_Y], dim=0), encode_only=False)  # concatenate data_X and data_Y for efficiency
            fake_Y = fake_Y_cat[:fake_Y_cat.shape[0]//2]
            feat_list = [feat_cat[:feat_cat.shape[0]//2]
                         for feat_cat in feat_list_cat]
            feat_list_id = [feat_cat[feat_cat.shape[0]//2:]
                            for feat_cat in feat_list_cat]

            # First train the generator and MLP_bundle
            self.discriminator.set_requires_grad(False)
            # Compute generator loss
            loss_G = self.compute_loss_G(fake_Y)

            # We need to finish setting up MLP_bundle.
            # This step is necessary (unless we hard code it) because the channel numbers can vary depending on which layers we use for computation of NCE loss
            if not self.setup:
                self.MLP_bundle.setup_MLP(feat_list)
                self.optimizer_MLP = torch.optim.Adam(
                    self.MLP_bundle.parameters(),
                    lr=0.0002, betas=(0.5, 0.999),
                )
                self.setup = True

            # We use flip_equivariance the same way as the original CUT: Randomly flip image inputted to generator, and also flip  the features extracted from the fake image
            # This is not the only way to implement flip equivariance.
            # For efficiency, we run the generator for input image again only if flip_bool is True.
            if self.cli_args.flip_equivariance:
                prob = 0.5
            else:
                prob = 1.0
            flip_bool = random.random() > prob
            if flip_bool:
                feat_list_tar_cat = self.generator(self.generator(torch.flip(
                    torch.cat([data_X, data_Y], dim=0), [3]), encode_only=False)[0], encode_only=True)
                feat_list_tar = [feat_cat[:feat_cat.shape[0]//2]
                                 for feat_cat in feat_list_tar_cat]
                feat_list_tar_id = [feat_cat[feat_cat.shape[0]//2:]
                                    for feat_cat in feat_list_tar_cat]
            else:
                feat_list_tar_cat = self.generator(
                    fake_Y_cat, encode_only=True)
                feat_list_tar = [feat_cat[:feat_cat.shape[0]//2]
                                 for feat_cat in feat_list_tar_cat]
                feat_list_tar_id = [feat_cat[feat_cat.shape[0]//2:]
                                    for feat_cat in feat_list_tar_cat]

            # compute MLP loss
            loss_MLP = self.compute_loss_MLP(
                feat_list, feat_list_tar, feat_list_id, feat_list_tar_id, flip_bool)

            # train generator and MLP
            self.optimizer_G.zero_grad()
            self.optimizer_MLP.zero_grad()
            (loss_G + loss_MLP).backward()
            self.optimizer_G.step()
            self.optimizer_MLP.step()

            # Next, we train the discriminator
            self.discriminator.set_requires_grad(True)
            fake_Y = self.ImgBufferY(fake_Y.detach())
            # compute discriminator loss
            loss_D = self.compute_loss_D(data_Y, fake_Y)
            # train discriminator
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

    def compute_loss_G(self, fake_Y):
        """ Compute the loss for generator. Here, we use LSGAN loss.
        Args:
            fake_Y (torch.tensor): generated mimicking distribution Y
        Returns:
            torch.tensor: the loss for generator
        """
        patch_fake = self.discriminator(fake_Y)
        loss_GAN = self.lossfn_GAN(patch_fake, torch.ones_like(patch_fake))
        return loss_GAN

    def compute_loss_D(self, data_Y, fake_Y):
        """ Compute the loss for discriminator.
        Args:
            data_Y (torch.tensor): data in the distribution Y
            fake_Y (torch.tensor): fake data produced by generatorY

        Returns:
            torch.tensor: the loss for discriminator
        """
        patch_real_Y = self.discriminator(data_Y)
        patch_fake_Y = self.discriminator(fake_Y)

        loss_real = self.lossfn_GAN(
            patch_real_Y, torch.ones_like(patch_real_Y))
        loss_fake = self.lossfn_GAN(
            patch_fake_Y, torch.zeros_like(patch_fake_Y))

        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_D

    def compute_loss_MLP(self, feat_list, feat_list_tar, feat_list_id, feat_list_tar_id, flip_bool):
        """ Compute NCE loss, used for training MLP and generator 
        Args: 
            feat_list (list): list of feature maps from encoder part of the generator for image in distribution X (features specified with --feat-str in self.cli_args)
            feat_list_tar (list): list of feature maps from encoder part of the generator for the generated image in distribution Y (input in distribution X)
            feat_list_id (list): list of feature maps from encoder part of the generator for image in distribution Y
            feat_list_tar_id (list): list of feature maps from encoder part of the generator for the generated image in distribution Y (input in distribution Y)
            flip_bool (bool): if True, enforce flip equivariance.
        Returns:
            torch.tensor: NCE loss for training MLP and generator
        """
        loss_MLP = 0.
        if flip_bool:
            feat_list_tar = [torch.flip(feat, [3]) for feat in feat_list_tar]
            feat_list_tar_id = [torch.flip(feat, [3])
                                for feat in feat_list_tar_id]
        # extract features for NCE loss with MLP
        out_feat_list, out_feat_list_tar = self.MLP_bundle(
            feat_list, feat_list_tar, num_patches=self.cli_args.num_patches)
        # extract features for identity (NCE) loss with MLP
        out_feat_list_id, out_feat_list_tar_id = self.MLP_bundle(
            feat_list_id, feat_list_tar_id, num_patches=self.cli_args.num_patches)
        # sum NCE loss for features extracted from the layers specified with --feat-str
        for out_feat, out_feat_tar, lossfn_NCE in zip(out_feat_list, out_feat_list_tar, self.lossfn_NCE_list):
            loss_MLP += lossfn_NCE(out_feat,
                                   out_feat_tar) * self.cli_args.NCE_ratio
        # sum identity (NCE) loss for features extracted from the layers specified with --feat-str
        for out_feat, out_feat_tar, lossfn_NCE in zip(out_feat_list_id, out_feat_list_tar_id, self.lossfn_NCE_list):
            loss_MLP += lossfn_NCE(out_feat,
                                   out_feat_tar) * self.cli_args.NCE_ratio_id
        return loss_MLP

    def visualize(self, mode, epoch):
        """ Save generated image.
        Args:
            mode (str): 'trn' or 'val'. The two options correspond to sampling and generating image from training/validation dataset.
            epoch (int): epoch at which to generate and save image. Note that the file name contains the epoch number.
        """
        if mode == 'trn':
            dlX = self.trn_dlX
            file_path = os.path.join(
                'visualize', self.cli_args.data_folder, 'trn')
        elif mode == 'val':
            dlX = self.val_dlX
            file_path = os.path.join(
                'visualize', self.cli_args.data_folder, 'val')
        else:
            file_path = os.path.join(
                'visualize', self.cli_args.data_folder, 'trn')
            logger.warning('warning: unreschecognized option' + str(mode))
            logger.warning('using mode=trn')
            dlX = self.trn_dlX

        os.makedirs(file_path, mode=0o755, exist_ok=True)

        for data_X in dlX:
            data_X = data_X.to(self.device)

            with torch.no_grad():
                img_batch_G, _ = self.generator(data_X, encode_only=False)
                data_X = (data_X + 1.0) / 2
                img_batch_G = (img_batch_G + 1.0) / 2
                torchvision.utils.save_image(data_X, os.path.join(
                    file_path, 'x' + str(epoch) + '.png'))
                torchvision.utils.save_image(img_batch_G, os.path.join(
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
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'MLP_bundle_state': self.MLP_bundle.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
            'optimizer_MLP_state': self.optimizer_MLP.state_dict(),
        }
        torch.save(state, os.path.join(file_path, 'CUT.state'))
        # logger.info('saving model')


if __name__ == '__main__':
    x = CUTTraining()
    x.main()
