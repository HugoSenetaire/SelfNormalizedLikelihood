import torch.nn as nn


def get_DC_GAN_Discriminator(input_size, cfg):
    return DCGANDiscriminator(
        in_channels=input_size[0],
        ngf=cfg.ngf,
        nout=cfg.nout,
        img_size=input_size[1],
        weight_norm=cfg.weight_norm,
    )


def get_BNDC_GAN_Discriminator(input_size, cfg):
    return BNDCGANDiscriminator(
        in_channels=input_size[0],
        ngf=cfg.ngf,
        nout=cfg.nout,
    )


def DCGANDiscriminator(in_channels=3, ngf=64, nout=1, img_size=32, weight_norm=False):
    """
    DCGAN Discriminator.
    """
    if img_size == 32 or img_size == 28:
        final_kernel = 2
    elif img_size == 64:
        final_kernel = 4
    else:
        raise ValueError
    if weight_norm:
        return nn.Sequential(
            # input is (nc) x 32 x 32
            nn.utils.weight_norm(nn.Conv2d(in_channels, ngf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.utils.weight_norm(nn.Conv2d(ngf, 2 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 8 x 8
            nn.utils.weight_norm(nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 4 x 4
            nn.utils.weight_norm(nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 2 x 2
            nn.utils.weight_norm(
                nn.Conv2d(8 * ngf, nout, final_kernel, 1, 0, bias=False)
            ),
            nn.Flatten(start_dim=1),
        )
    else:
        return nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(in_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.Conv2d(ngf, 2 * ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 8 x 8
            nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 4 x 4
            nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 2 x 2
            nn.Conv2d(8 * ngf, nout, final_kernel, 1, 0, bias=False),
            nn.Flatten(start_dim=1),
        )


def BNDCGANDiscriminator(in_channels=3, ngf=64, nout=1):
    """
    DCGAN Discriminator with batchnorm.
    """
    return nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(in_channels, ngf, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf) x 16 x 16
        nn.Conv2d(ngf, 2 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(2 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*2) x 8 x 8
        nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(4 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*4) x 4 x 4
        nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(8 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*8) x 2 x 2
        nn.Conv2d(8 * ngf, nout, 2, 1, 0),
        nn.Flatten(start_dim=1),
    )
