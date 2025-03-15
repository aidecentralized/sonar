import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, img_size: int = 32, nc: int = 3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self) -> "Generator":
        clone = Generator(
            self.params[0], self.params[1], self.params[2], self.params[3]
        )
        clone.load_state_dict(self.state_dict())
        return clone.to(self.device)

    # def shuffle_filter(self):
    #     s1 = torch.randperm(self.params[1] * 2)
    #     s2 = torch.randperm(self.params[1])
    #     s3 = torch.randperm(self.params[3])
    #     self.conv_blocks[2].weight = torch.nn.Parameter(self.conv_blocks[2].weight[s1, :, :, :])
    #     self.conv_blocks[6].weight = torch.nn.Parameter(self.conv_blocks[6].weight[s2, :, :, :])
    #     self.conv_blocks[9].weight = torch.nn.Parameter(self.conv_blocks[9].weight[s3, :, :, :])


class DeepGenerator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, img_size: int = 224, nc: int = 3):
        super(DeepGenerator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 32
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            # nn.Conv2d(nz, ngf, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(0.2, inplace=True),
            # 7x7
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(nz, 2 * ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ngf, 2 * ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56x56
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 x 112
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 224 x 224
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # out = self.l1(z)
        # out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(z)
        return img

    # return a copy of its own
    def clone(self, copy_params: bool = True) -> "DeepGenerator":
        clone = DeepGenerator(
            self.params[0], self.params[1], self.params[2], self.params[3]
        )
        if copy_params:
            clone.load_state_dict(self.state_dict())
        return clone.to(self.device)


class DCGAN_Generator(nn.Module):
    """Generator from DCGAN: https://arxiv.org/abs/1511.06434"""

    def __init__(
        self,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 3,
        img_size: Union[int, List[int], Tuple[int, int]] = 64,
        slope: float = 0.2,
    ):
        super(DCGAN_Generator, self).__init__()
        self.nz = nz
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid(),
            # nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class DCGAN_CondGenerator(nn.Module):
    """Generator from DCGAN: https://arxiv.org/abs/1511.06434"""

    def __init__(
        self,
        num_classes: int,
        nz: int = 100,
        n_emb: int = 50,
        ngf: int = 64,
        nc: int = 3,
        img_size: Union[int, List[int], Tuple[int, int]] = 64,
        slope: float = 0.2,
    ):
        super(DCGAN_CondGenerator, self).__init__()
        self.nz = nz
        self.emb = nn.Embedding(num_classes, n_emb)
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz + n_emb, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x
            nn.Conv2d(ngf, nc, 3, 1, 1),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.emb(y)
        z = torch.cat([z, y], dim=1)
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc: int = 3, img_size: int = 32):
        super().__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, bn: bool = True
        ) -> nn.ModuleList:
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return nn.ModuleList(block)

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)
