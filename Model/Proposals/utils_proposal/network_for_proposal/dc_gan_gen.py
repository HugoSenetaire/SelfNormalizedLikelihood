import torch.nn as nn


def get_DCGANGenerator(input_size, cfg):
    return DCGANGenerator(
        cfg.noise_dim,
        cfg.activation,
        out_channels=input_size[0],
        ngf=cfg.ngf,
        img_size=input_size[1],
    )


def DCGANGenerator(noise_dim, activation=None, out_channels=3, ngf=64, img_size=32):
    """
    DCGan Generator.
    """

    class G(nn.Module):
        """
        Generator torch module.
        """

        def __init__(self, activation=None):
            super().__init__()
            self.img_size = img_size
            if activation == "sigmoid":
                final_act = nn.Sigmoid()
            elif activation == "tanh":
                final_act = nn.Tanh()
            else:
                final_act = None
            print(img_size)

            if img_size == 32:
                first_kernel = 2
            elif img_size == 64:
                first_kernel = 4
            else:
                raise ValueError
            self.first = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(noise_dim, ngf * 8, first_kernel, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1),
            )
            self.activation = final_act

        def forward(self, x):
            """
            Forward pass.
            """
            x = x.view(x.size(0), -1, 1, 1)
            x = self.first(x)
            if self.activation is not None:
                x = self.activation(x)
            return x

    return G(
        activation,
    )
