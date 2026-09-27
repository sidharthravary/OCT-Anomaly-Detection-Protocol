"""nets.py - network definitions and losses shared by the training / scoring scripts.

ConvAutoencoder (M1, PRD 05), ConvVAE (M3, PRD 09) and a differentiable SSIM.
Inputs are 1 x 224 x 224 in [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENC_CHANNELS = (32, 64, 128, 256, 256)   # 224 -> 112 -> 56 -> 28 -> 14 -> 7


def conv_block(c_in, c_out):
    """Conv 3x3 (stride 2) -> BatchNorm -> LeakyReLU."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))


def deconv_block(c_in, c_out):
    """ConvTranspose 4x4 (stride 2) -> BatchNorm -> LeakyReLU; doubles the spatial size."""
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        chans = (in_ch,) + ENC_CHANNELS
        self.blocks = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])])

    def forward(self, x):
        return self.blocks(x)                       # B x 256 x 7 x 7


class Decoder(nn.Module):
    """Mirror of Encoder: 256 x 7 x 7 -> 1 x 224 x 224, sigmoid output."""

    def __init__(self, out_ch=1):
        super().__init__()
        chans = ENC_CHANNELS[::-1]                  # 256, 256, 128, 64, 32
        self.blocks = nn.Sequential(*[deconv_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.out = nn.ConvTranspose2d(chans[-1], out_ch, 4, stride=2, padding=1)

    def forward(self, h):
        return torch.sigmoid(self.out(self.blocks(h)))


class ConvAutoencoder(nn.Module):
    """M1: 5-block encoder, 1x1 bottleneck to 64 channels (7 x 7 x 64), mirrored decoder."""

    def __init__(self, bottleneck=64):
        super().__init__()
        self.encoder = Encoder()
        self.to_code = nn.Conv2d(ENC_CHANNELS[-1], bottleneck, 1)
        self.from_code = nn.Sequential(nn.Conv2d(bottleneck, ENC_CHANNELS[-1], 1), nn.LeakyReLU(0.2, inplace=True))
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.from_code(self.to_code(self.encoder(x))))


class ConvVAE(nn.Module):
    """M3: same encoder / decoder as M1, 256-d latent via fully connected layers from the 7 x 7 map."""

    def __init__(self, latent=256):
        super().__init__()
        flat = ENC_CHANNELS[-1] * 7 * 7
        self.encoder = Encoder()
        self.fc_mu = nn.Linear(flat, latent)
        self.fc_logvar = nn.Linear(flat, latent)
        self.fc_dec = nn.Sequential(nn.Linear(latent, flat), nn.LeakyReLU(0.2, inplace=True))
        self.decoder = Decoder()

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, ENC_CHANNELS[-1], 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decode(z), mu, logvar


def kl_divergence(mu, logvar):
    """Per-sample KL(q(z|x) || N(0, I)), summed over latent dims."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


# ------------------------------------------------------------ Deep SVDD
class ResNetFeatures(nn.Module):
    """ImageNet ResNet18 up to layer4; returns the layer3 (256 x 14 x 14) and layer4 (512 x 7 x 7) maps.
    Expects 3 x 224 x 224 input normalised with ImageNet mean / std."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        l3 = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return l3, self.layer4(l3)


def frozen_embedding(l3, l4):
    """svdd_frozen: GAP(layer3) ++ GAP(layer4) -> 768-d, L2-normalised."""
    z = torch.cat([l3.mean(dim=(2, 3)), l4.mean(dim=(2, 3))], dim=1)
    return F.normalize(z, dim=1)


class DeepSVDD(nn.Module):
    """svdd_ft: ResNet18 backbone + bias-free linear head (GAP(layer4) 512 -> 128).

    Everything up to and including layer2 is frozen. BatchNorm layers are kept in eval mode with frozen
    affine parameters: their shifts are bias terms, which Deep SVDD must avoid (Ruff et al. 2018) because
    they let the network map every input to c.
    """

    def __init__(self, rep_dim=128, pretrained=True):
        super().__init__()
        self.features = ResNetFeatures(pretrained)
        self.head = nn.Linear(512, rep_dim, bias=False)
        for mod in (self.features.stem, self.features.layer1, self.features.layer2):
            for p in mod.parameters():
                p.requires_grad = False
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self.features.modules():   # BN always uses the ImageNet running statistics
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        _, l4 = self.features(x)
        return self.head(l4.mean(dim=(2, 3)))


# ------------------------------------------------------------------ MKD
# Critical layers: last ReLU of VGG-16 blocks 2-5 (relu2_2, relu3_3, relu4_3, relu5_3).
MKD_TAPS = (8, 15, 22, 29)                     # indices in torchvision vgg16().features
MKD_TAP_CHANNELS = (128, 256, 512, 512)


class VGGTeacher(nn.Module):
    """Frozen ImageNet VGG-16; returns activations at the MKD critical layers."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:MKD_TAPS[-1] + 1].eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in MKD_TAPS:
                outs.append(x)
        return outs


class VGGStudent(nn.Module):
    """Smaller VGG-style student (Salehi et al. 2021): thin intermediate convs, but each tapped block ends in
    a conv with the teacher's channel count so activations can be compared one-to-one."""

    CFG = ((16, 64), (16, 128), (16, 16, 256), (16, 16, 512), (16, 16, 512))

    def __init__(self):
        super().__init__()
        blocks, c_in = [], 3
        for b, chans in enumerate(self.CFG):
            layers = [] if b == 0 else [nn.MaxPool2d(2)]
            for c in chans:
                layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(inplace=True)]
                c_in = c
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for b, block in enumerate(self.blocks):
            x = block(x)
            if b >= 1:                       # blocks 2-5 are the tapped ones
                outs.append(x)
        return outs


def mkd_loss(student_feats, teacher_feats, lam=0.01):
    """Per-sample L_val (mean squared distance) + lam * L_dir (1 - cosine), summed over the critical layers.
    Computed in float32 even when the features come from a mixed-precision forward pass."""
    student_feats = [f.float() for f in student_feats]
    teacher_feats = [f.float() for f in teacher_feats]
    val = sum(((s - t) ** 2).flatten(1).mean(1) for s, t in zip(student_feats, teacher_feats))
    dirn = sum(1 - F.cosine_similarity(s.flatten(1), t.flatten(1), dim=1) for s, t in zip(student_feats, teacher_feats))
    return val + lam * dirn


def mkd_map(student_feats, teacher_feats, size):
    """Anomaly map: per-location squared discrepancy, normalised per layer, upsampled and summed."""
    total = 0
    for s, t in zip(student_feats, teacher_feats):
        s, t = s.float(), t.float()
        d = ((s - t) ** 2).mean(1, keepdim=True)
        total = total + F.interpolate(d / (d.flatten(1).mean(1).view(-1, 1, 1, 1) + 1e-8), size=size,
                                      mode="bilinear", align_corners=False)
    return total[:, 0]


# ------------------------------------------------------------------ SSIM
def _gaussian_window(size=11, sigma=1.5, device=None):
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, data_range=1.0):
    """Per-pixel SSIM between single-channel batches x, y in [0, data_range]."""
    w = _gaussian_window(device=x.device)
    pad = w.shape[-1] // 2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = F.conv2d(x, w, padding=pad), F.conv2d(y, w, padding=pad)
    sxx = F.conv2d(x * x, w, padding=pad) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim(x, y):
    """Mean SSIM per sample (B,)."""
    return ssim_map(x, y).flatten(1).mean(dim=1)
