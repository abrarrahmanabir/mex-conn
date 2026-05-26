import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature recalibration."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualDoubleConv(nn.Module):
    """Residual Double Conv with SE and Dropout."""
    def __init__(self, in_ch, out_ch, p_drop=0.2, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.use_se = use_se
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
        self.dropout = nn.Dropout2d(p_drop)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x + residual

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, features=[64, 128, 256, 512, 1024], p_drop=0.2, use_se=True):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            self.layers.append(ResidualDoubleConv(prev_ch, f, p_drop=p_drop, use_se=use_se))
            prev_ch = f
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feats = []
        for conv in self.layers:
            x = conv(x)
            feats.append(x)
            x = self.pool(x)
        return feats

class UNetDecoder(nn.Module):
    def __init__(self, features=[1024, 512, 256, 128, 64], out_ch=1, p_drop=0.2, use_se=True):
        super().__init__()
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(nn.ConvTranspose2d(features[i], features[i+1], 2, stride=2))
            self.dec_convs.append(
                ResidualDoubleConv(features[i+1] + features[i+1], features[i+1], p_drop=p_drop, use_se=use_se)
            )
        self.out_conv = nn.Conv2d(features[-1], out_ch, 1)
    def forward(self, feats):
        x = feats[-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = feats[-2 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:])
            x = torch.cat([skip, x], dim=1)
            x = self.dec_convs[i](x)
        return self.out_conv(x)

class MultiHeadUNet(nn.Module):

    def __init__(self, in_ch=1, base_features=[64, 128, 256, 512, 1024], out_ch_per_head=1, num_heads=3, p_drop=0.2, use_se=True):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, features=base_features, p_drop=p_drop, use_se=use_se)
        dec_feats = base_features[::-1]
        self.decoders = nn.ModuleList([
            UNetDecoder(features=dec_feats, out_ch=out_ch_per_head, p_drop=p_drop, use_se=use_se) for _ in range(num_heads)
        ])
    def forward(self, x):
        feats = self.encoder(x)
        outs = []
        for dec in self.decoders:
            outs.append(dec(feats))  # Each out: (B, 1, H, W)
        return torch.cat(outs, dim=1)  # (B, 3, H, W)
