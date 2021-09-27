import torch.nn as nn
import torch


class discriminator(nn.Module):
    def __init__(self, in_channels=6):
        """ Initiate the discriminator model in pix2pix. 
        Args:
            in_channels (int): the number of channesl in the image x 2. For color images, this is 6. 
        """
        super().__init__()
        model_list = nn.ModuleList()
        model_list.append(ConvBlock(in_channels, 64, batch_norm=False, bias=True))
        model_list.append(ConvBlock(64, 128, batch_norm=True, bias=False))
        model_list.append(ConvBlock(128, 256, batch_norm=True, bias=False))
        # In the paper, the authors say that all strides are 2 in their network structure
        # (C64-C128-C256-C512)
        # However, to get receptive field of 70 for the final pixel, the stride for C512 should be 1
        # This is also what is actually implemented in their code.
        model_list.append(ConvBlock(256, 512, batch_norm=True, bias=False, stride=1))
        model_list.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True))
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
    def __init__(self, in_channels=3, out_channels=3):
        """ Initiate the generator model in pix2pix. 
        Args:
            in_channels (int): the number of channels in the input image. If the input is colored image, this is 3.
            out_channels (int): the number of channels in the output image. If the output is colored image, this is 3.
        """

        super().__init__()
        self.encoder_list = nn.ModuleList()
        self.encoder_list.append(ConvBlock(in_channels, 64, batch_norm=False, bias=True))
        self.encoder_list.append(ConvBlock(64, 128, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(128, 256, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(256, 512, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(512, 512, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(512, 512, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(512, 512, batch_norm=True, bias=False))
        self.encoder_list.append(ConvBlock(512, 512, batch_norm=True, bias=False))
        # In the code, the innermost layer has uses ReLU instead of leaky RelU
        # This requires that I use final=True in the last layer.
        # However, I stick with the design in the paper.

        self.decoder_list = nn.ModuleList()
        self.decoder_list.append(ConvTransBlock(512, 512, dropout=True))
        self.decoder_list.append(ConvTransBlock(1024, 512, dropout=True))
        self.decoder_list.append(ConvTransBlock(1024, 512, dropout=False))
        self.decoder_list.append(ConvTransBlock(1024, 512, dropout=False))
        self.decoder_list.append(ConvTransBlock(1024, 256, dropout=False))
        self.decoder_list.append(ConvTransBlock(512, 128, dropout=False))
        self.decoder_list.append(ConvTransBlock(256, 64, dropout=False))
        self.decoder_list.append(ConvTransBlock(128, 3, final=True, dropout=False))

        self._initialize_params()

    def forward(self, x):
        # In the paper, the layers are Convolution-BatchNorm-ReLU
        # However, in their code, the layers are ReLU-Conv-BatchNorm
        # This makes the skip connections slightly different.
        # I stick with the design in the paper, where the authors explicitly state
        # "skip connections concatenate activations from layer i to layer n-i" and
        # "Ck denotes Convolution-BatchNorm-ReLU layer with k-filters" and
        # "CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with k-filters"
        # This difference does not seem to significantly affect the performance. 

        encoder_result_list = []
        encoder_result = x
        for i in range(len(self.encoder_list) - 1):
            encoder_result = self.encoder_list[i](encoder_result)
            encoder_result_list.insert(0, encoder_result)
        encoder_result = self.encoder_list[-1](encoder_result)

        decoder_result = self.decoder_list[0](encoder_result)
        for decoder_block, encoder_result in zip(self.decoder_list[1:], encoder_result_list):
            decoder_result = decoder_block(torch.cat((decoder_result, encoder_result), dim=1))

        return decoder_result

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
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, batch_norm=True, bias=False, final=False):
        """ The convolution block in pix2pix
        Args:
            in_size (int): number of input features
            out_size (int): number of output features
            kernel_size (int):  kernel size of convolutioon.
            stride (int): stride size of convolution
            padding (int): padding size of convolution
            batch_norm (bool): if True, use batch norm.
            bias (bool): if False, switch off bias in convolution (if we use batch norm, bias is not necessary)
            final (bool): set it to True for final convolution block in generator if it is desired to make the model 
                architecture the same as in the pytorch code provided by the authors

        """
        super().__init__()
        ConvBlockList = nn.ModuleList()
        ConvBlockList.append(nn.Conv2d(in_size, out_size,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias)
                             )
        if batch_norm:
            ConvBlockList.append(nn.BatchNorm2d(out_size))
        if not final:
            ConvBlockList.append(nn.LeakyReLU(0.2))
        else:
            ConvBlockList.append(nn.ReLu())
        self.model = nn.Sequential(*ConvBlockList)

    def forward(self, x):
        return self.model(x)


class ConvTransBlock(nn.Module):
    """ The transpose convolution block in pix2pix
    Args:
        in_size (int): number of input features
        out_size (int): number of output features
        dropout (bool): use dropout if True.
        kernel_size (int):  kernel size of convolutioon.
        stride (int): stride size of convolution
        padding (int): padding size of convolution
        final (bool): set it to True for final transpose convolution layer.
    """
    def __init__(self, in_size, out_size, dropout=False, kernel_size=4, stride=2, padding=1, final=False):
        super().__init__()
        ConvTransBlockList = nn.ModuleList()
        ConvTransBlockList.append(nn.ConvTranspose2d(in_size, out_size,
                                                     kernel_size=kernel_size, stride=2,
                                                     padding=padding, bias=False)
                                  )
        if not final:
            ConvTransBlockList.append(nn.BatchNorm2d(out_size))

        if not final:
            ConvTransBlockList.append(nn.ReLU())
        else:
            ConvTransBlockList.append(nn.Tanh())

        if dropout:
            ConvTransBlockList.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*ConvTransBlockList)

    def forward(self, x):
        return self.model(x)
