import torch.nn as nn
import torch
import torchvision.transforms as transforms


class discriminator(nn.Module):
    def __init__(self, in_channels=3):
        """ Initiate the discriminator model in cycleGAN.
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
    def __init__(self, in_channels=3, out_channels=3, n_layers=6):
        """ Initialize the generator model.

        Args:
            in_channels (int): the number of channels in the input image. If the input is colored image, this is 3.
            out_channels (int): the number of channels in the output image. If the output is colored image, this is 3.
            n_layers (int): the number of resnet layers.
        """

        super().__init__()
        generator_list = nn.ModuleList()
        # c7s1-64
        generator_list.append(nn.ReflectionPad2d(3))
        generator_list.append(ConvBlock(
            in_channels, 64, kernel_size=7, stride=1, padding=0,
            leaky=False, instance_norm=True, bias=False
        ))
        # d128, d256 (down-sampling layers)
        generator_list.append(ConvBlock(
            64, 128, kernel_size=3, stride=2, padding=1,
            leaky=False, instance_norm=True, bias=False
        ))
        generator_list.append(ConvBlock(
            128, 256, kernel_size=3, stride=2, padding=1,
            leaky=False, instance_norm=True, bias=False
        ))
        # resblocks
        for n in range(n_layers):
            generator_list.append(ResnetBlock(256))
        # u128, u64 (up-sampleing layers)
        generator_list.append(ConvTransBlock(256, 128))
        generator_list.append(ConvTransBlock(128, 64))
        # c7s1-3
        generator_list.append(nn.ReflectionPad2d(3))
        generator_list.append(nn.Conv2d(64, out_channels,
                                        kernel_size=7, padding=0)
                              )
        generator_list.append(nn.Tanh())
        # define generator model
        self.model = nn.Sequential(*generator_list)
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
        """ Switch on and off the requires_grad option for the parameters in the generator.
        Args:
            requires_grad (bool): if set to True, the parameters of the generator requires gradient, and vice versa.
        """
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad


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


class down_sampler(nn.Module):
    """ Downsample image
    """

    def __init__(self, blur=True, pool_size=2):
        """
        Args:
            blur (bool): blur / don't blur
            pool_size (int): pooling size
        """
        super().__init__()
        self.blur = blur
        if self.blur:
            self.padding = nn.ReflectionPad2d(3)
            self.blur = transforms.GaussianBlur(7, sigma=3)
        self.pool = torch.nn.AvgPool2d(pool_size)

    def forward(self, x):
        out = x
        if self.blur:
            out = self.padding(out)
            out = self.blur(out)
        out = self.pool(out)
        return out
