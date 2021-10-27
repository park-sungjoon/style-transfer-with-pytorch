import torch.nn as nn
import torch
import torchvision.transforms as transforms
import math


class discriminator(nn.Module):
    def __init__(self, in_channels=3):
        """ Initiate the discriminator model in CUT (same as discriminator in cycleGAN).
        The basic structure is the same as in pix2pix. However, the in_channels
        should be the number of channels in the image, in contrast to the
        discriminator in pix2pix, where the in_channels is double this number.

        Args:
            in_channels (int): the number of channesl in the image.
        """
        super().__init__()
        model_list = nn.ModuleList()
        model_list.append(
            ConvBlock(in_channels, 64, leaky=True, instance_norm=False, bias=True))
        model_list.append(ConvBlock(64, 128, leaky=True,
                                    instance_norm=True, bias=False))
        model_list.append(ConvBlock(128, 256, leaky=True,
                                    instance_norm=True, bias=False))
        model_list.append(ConvBlock(256, 512, leaky=True,
                                    instance_norm=True, bias=False, stride=1))
        model_list.append(nn.Conv2d(512, 1, kernel_size=4,
                                    stride=1, padding=1, bias=True))
        self.model = nn.Sequential(*model_list)

        self._initialize_params()

    def forward(self, x):
        return self.model(x)

    def _initialize_params(self):
        for m in self.modules():
            if type(m) in {
                nn.Conv2d, nn.ConvTranspose2d
            }:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def set_requires_grad(self, requires_grad):
        """ Switch on and off the requires_grad option for the parameters in the discriminator.
        Args:
            requires_grad (bool): if set to True, the parameters of the discriminator requires gradient, and vice versa.
        """
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad


class generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_layers=9, feat_str_list=['pix', 'd128', 'd256', 'R256_1', 'R256_5']):
        """ Initialize the generator model (same structure as cycleGAN).
        The generator has the following structure (Refer to cycle GAN paper for notation):
            c7s1-64
            d128, d256 (down-sampling layers)
            R256 x n_layers (Resnet blocks)
            u128, u64 (up-sampleing layers)
            c7s1-3

        Args:
            in_channels (int): the number of channels in the input image. If the input is colored image, this is 3.
            out_channels (int): the number of channels in the output image. If the output is colored image, this is 3.
            n_layers (int): the number of resnet layers.
            feat_str_list (list): list of strings denoting features to be extracted for contrastive learning.
                Should take values in : 'pix', 'c7s1_64', 'd128', 'd256', 'R256_n'.
                Note that for R256_n, n takes maximum value of n_layers/2 (rounded up)
        """
        super().__init__()
        self.n_layers = n_layers
        self.feat_str_list = feat_str_list
        self.num_feat = len(feat_str_list)

        # run some simple checks
        assert len(feat_str_list) == len(set(feat_str_list)
                                         ), 'cannot select same feature more than once'
        possible_feat = ['pix', 'c7s1_64', 'd128', 'd256'] + \
            [f'R256_{n}' for n in range(1, math.ceil(n_layers / 2) + 1)]
        for feat_str in feat_str_list:
            assert feat_str in possible_feat, f'unrecognized feature {feat_str}'

        self.c7s1_64 = nn.Sequential(nn.ReflectionPad2d(3),
                                     ConvBlock(in_channels, 64, kernel_size=7,
                                               stride=1, padding=0, leaky=False,
                                               instance_norm=True, bias=False),
                                     )
        self.d128 = ConvBlock(64, 128, kernel_size=3,
                              stride=2, padding=1, leaky=False,
                              instance_norm=True, bias=False,
                              )
        self.d256 = ConvBlock(128, 256, kernel_size=3,
                              stride=2, padding=1, leaky=False,
                              instance_norm=True, bias=False,
                              )
        for n in range(1, n_layers + 1):
            setattr(self, f'R256_{n}', ResnetBlock(256))
        self.u128 = ConvTransBlock(256, 128)
        self.u64 = ConvTransBlock(128, 64)
        self.c7s1_3 = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(64, out_channels,
                                              kernel_size=7, padding=0),
                                    )
        self.Tanh = nn.Tanh()
        self._initialize_params()

    def forward(self, pix, encode_only):
        """ 
        Args:
            pix (torch.tensor): batch of data in distribution X
            encode_only (bool): if True, used only for encoding 
        Returns:
            if encode_only: list of feature maps
            else: (generated image in distribution Y, list of feature maps)
        """
        feat_list = []
        if 'pix' in self.feat_str_list:
            feat_list.append(pix)
            if encode_only and len(feat_list) == self.num_feat:
                return feat_list

        out = self.c7s1_64(pix)
        if 'c7s1_64' in self.feat_str_list:
            feat_list.append(out)
            if encode_only and len(feat_list) == self.num_feat:
                return feat_list

        out = self.d128(out)
        if 'd128' in self.feat_str_list:
            feat_list.append(out)
            if encode_only and len(feat_list) == self.num_feat:
                return feat_list

        out = self.d256(out)
        if 'd256' in self.feat_str_list:
            feat_list.append(out)
            if encode_only and len(feat_list) == self.num_feat:
                return feat_list

        for n in range(1, self.n_layers + 1):
            out = getattr(self, f'R256_{n}')(out)
            if f'R256_{n}' in self.feat_str_list:
                feat_list.append(out)
                if encode_only and len(feat_list) == self.num_feat:
                    return feat_list
        out = self.u128(out)
        out = self.u64(out)
        out = self.c7s1_3(out)
        return out, feat_list

    def _initialize_params(self):
        for m in self.modules():
            if type(m) in {
                nn.Conv2d, nn.ConvTranspose2d
            }:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def set_requires_grad(self, requires_grad):
        """ Switch on and off the requires_grad option for the parameters in the generator.
        Args:
            requires_grad (bool): if set to True, the parameters of the generator requires gradient, and vice versa.
        """
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad


class MLP_bundle(nn.Module):
    """ 
    Collection of MLP used to extract features for NCE loss from feature maps of generator
    Note that after defining an instance of MLP_bundle, we need to finish setting it up by calling:
        1. set_device: which device to load MLP
        2. setup_MLP: define list of MLP
    """

    def __init__(self, out_size=256):
        """
        Args:
            out_size (int): size of feature extracted with MLP
        """
        super().__init__()
        self.out_size = out_size

    def set_device(self, device=None):
        """ Sets device to which we load MLP
        Args:
            device (torch.device): torch.device("cuda" or "cpu"). If not specified, detects whether cuda is available and automatically sets device.
        """
        if device == None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = device

    def setup_MLP(self, feat_list):
        """Define list of MLP.
        Args:
            feat_list (list): list of feature maps extracted from generator. We need the number of channels in the feature maps. 
        """
        self.MLP_list = nn.ModuleList()
        for feat in feat_list:
            in_size = feat.shape[1]
            MLP = nn.Sequential(*[nn.Linear(in_size, self.out_size),
                                  nn.ReLU(),
                                  nn.Linear(self.out_size, self.out_size)]
                                ).to(self.device)
            self.MLP_list.append(MLP)
        self._initialize_params()

    def _initialize_params(self,):
        for m in self.modules():
            if type(m) in {
                nn.Linear
            }:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def normalize(self, unnorm_t):
        """ Normalize along the first dimension
        Args:
            unnorm_t (torch.tensor): unnormalized tensor of the form (batch, out_size)
        Returns:
            torch.tensor: normalized tensor.
        """
        norm = (unnorm_t**2).sum(dim=1, keepdim=True)**(0.5)
        norm_t = unnorm_t / (norm + 1e-7)
        return norm_t

    def forward(self, feat_list, gen_feat_list, num_patches=256):
        """ 
        Args:
            feat_list: features extracted from generator(data_X/Y)
            gen_feat_list: features extracted from generator(generator(data_X/Y))
            num_patches (int): number of patches to sample from
        returns:
            tuple of list of features extracted after passing the feature maps through MLP, for feat_list and gen_feat_list.
        """
        out_feat_list = []
        out_gen_feat_list = []
        assert len(feat_list) == len(gen_feat_list) and len(gen_feat_list) == len(
            self.MLP_list), f'{len(feat_list)}, {len(gen_feat_list)}, {len(self.MLP_list)}'
        for feat, gen_feat, MLP in zip(feat_list, gen_feat_list, self.MLP_list):
            feat = feat.permute(0, 2, 3, 1).flatten(1, 2)
            gen_feat = gen_feat.permute(0, 2, 3, 1).flatten(1, 2)
            patch_idx = torch.randperm(feat.shape[1], device=self.device)[
                :min(num_patches, feat.shape[1])]
            feat_sample = feat[:, patch_idx, :].flatten(0, 1)
            gen_feat_sample = gen_feat[:, patch_idx, :].flatten(0, 1)
            out = MLP(torch.cat([feat_sample, gen_feat_sample], dim=0))
            out = self.normalize(out)
            out_feat_list.append(out[0:feat_sample.shape[0], :])
            out_gen_feat_list.append(out[feat_sample.shape[0]:, :])
        return out_feat_list, out_gen_feat_list


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, leaky=False, instance_norm=True, bias=False):
        """ The convolution block in cycleGAN
        Args:
            in_size (int): number of input features
            out_size (int): number of output features
            kernel_size (int):  kernel size of convolutioon.
            stride (int): stride size of convolution
            padding (int): padding size of convolution
            leaky (bool): whether to use leaky relu or plain relu.
            instance_norm (bool): if True, use instance norm.
            bias (bool): if False, switch off bias in convolution (if we use instance norm, bias is not necessary)
        """
        super().__init__()
        ConvBlockList = nn.ModuleList()
        ConvBlockList.append(nn.Conv2d(in_size, out_size,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias)
                             )
        if instance_norm:
            ConvBlockList.append(nn.InstanceNorm2d(out_size))
        if leaky:
            ConvBlockList.append(nn.LeakyReLU(0.2))
        else:
            ConvBlockList.append(nn.ReLU())
        self.model = nn.Sequential(*ConvBlockList)

    def forward(self, x):
        return self.model(x)


class ConvTransBlock(nn.Module):
    """ The transpose convolution block in cycleGAN
    """

    def __init__(self, in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1):
        """ Initialize the transpose convolution block
        Args:
            in_size (int): number of input features
            out_size (int): number of output features
            kernel_size (int):  kernel size of convolutioon.
            stride (int): stride size of convolution
            padding (int): padding size of convolution
            output_padding(int): padding size of output
        """
        super().__init__()
        ConvTransBlockList = nn.ModuleList()
        ConvTransBlockList.append(nn.ConvTranspose2d(in_size, out_size,
                                                     kernel_size=kernel_size, stride=stride,
                                                     padding=padding, output_padding=output_padding,
                                                     bias=False)
                                  )
        ConvTransBlockList.append(nn.InstanceNorm2d(out_size))
        ConvTransBlockList.append(nn.ReLU())
        self.model = nn.Sequential(*ConvTransBlockList)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    """ ResnetBlock in cycleGAN generator
    """

    def __init__(self, io_dim):
        """" Create the convolutional block.
        Residal connection is implemented in the forward function.

        Args:
            io_dim (int): dimension of input, which is equal to the dimension of the output.
        """
        super().__init__()
        ResnetBlock_list = nn.ModuleList()
        ResnetBlock_list.append(nn.ReflectionPad2d(1))
        ResnetBlock_list.append(nn.Conv2d(io_dim, io_dim, kernel_size=3,
                                          padding=0, bias=False)
                                )
        ResnetBlock_list.append(nn.InstanceNorm2d(io_dim))
        ResnetBlock_list.append(nn.ReLU())
        ResnetBlock_list.append(nn.ReflectionPad2d(1))
        ResnetBlock_list.append(nn.Conv2d(io_dim, io_dim, kernel_size=3,
                                          padding=0, bias=False)
                                )
        ResnetBlock_list.append(nn.InstanceNorm2d(io_dim))
        self.module = nn.Sequential(*ResnetBlock_list)

    def forward(self, x):
        out = x + self.module(x)
        return out


class NCE_loss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, feat_src, feat_tar, tau=0.07):
        """
        basically copy of the pseudocode provided by authors in CUT paper.
        feat_src: sampled features from images in the source domain
        feat_tar: sampled features from generated images in the target domain
        """
        feat_src = feat_src.detach()
        # batch_size * num_patches (batch size outputted from MLP)
        MLPbatch_size = feat_src.shape[0]
        MLPfeat_size = feat_src.shape[1]  # size of features outputted by MLP
        num_patches = MLPbatch_size // self.batch_size
        # positive logits
        logits_pos = (feat_src * feat_tar).sum(dim=1, keepdim=True) / tau

        # negative logits
        logits_neg = torch.bmm(feat_tar.view(self.batch_size, -1, MLPfeat_size),
                               feat_src.view(self.batch_size, -1, MLPfeat_size).transpose(1, 2)) / tau
        # remove redundant terms (overlaps with positive logits)
        identity_matrix = torch.eye(
            num_patches, device=feat_src.device, dtype=torch.bool)[None, :, :]
        logits_neg.masked_fill_(identity_matrix, -10.0)
        logits_neg = logits_neg.view(-1, num_patches)
        # gather logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        # the NCE loss is just the cross entropy loss with the positive as target
        loss = self.loss_fn(logits, torch.zeros(
            logits.shape[0], dtype=torch.long, device=feat_src.device))
        return loss
