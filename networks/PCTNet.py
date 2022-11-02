import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class CMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                        groups=1, bias=False),

            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SE(inp, hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class NoBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = nn.Conv2d(cin, cmid, kernel_size=1, stride=1, padding=0, bias=False)

        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = nn.Conv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False)  # Original code has it on conv1!!

        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)

        self.conv3 = nn.Conv2d(cmid, cout, kernel_size=1, stride=1, padding=0, bias=False)

        self.gelu = nn.GELU()

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)
        # self.normal = nn.BatchNorm2d(cin)
        # self.normal = nn.GroupNorm(32, cin, eps=1e-6)
        self.normal = nn.GroupNorm(1, cin, eps=1e-6)

    def forward(self, x):
        # Residual branch
        residual = x
        # x = self.normal(x)
        if (self.stride != 1 or self.cin != self.cout):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        x = self.normal(x)


        # Unit's branch
        y = self.gelu(self.gn1(self.conv1(x)))  # 1X1

        out_to_trans = self.gelu(self.gn2(self.conv2(y)))  # 3X3

        y = self.conv3(y)
        y = self.gn3(y)
        y = self.gelu(y + residual)
        return y, out_to_trans

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        heads = inp // dim_head
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid([torch.arange(self.ih), torch.arange(self.iw)])  # , indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Conv2d(inp, oup, 1),
            nn.Dropout2d(dropout, inplace=True)
        ) if project_out else nn.Identity()

        self.projQ = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )
        self.projK = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )

        self.projV = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )

    def forward(self, x, y=None):
        q = self.projQ(x)
        k = self.projK(x)
        v = self.projV(x)

        q = rearrange(q, 'b c ih iw -> b (ih iw) c')
        k = rearrange(k, 'b c ih iw -> b (ih iw) c')
        v = rearrange(v, 'b c ih iw -> b (ih iw) c')

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2))
        if y is not None:
            q_y = self.projQ(y)
            k_y = self.projK(y)
            q_y = rearrange(q_y, 'b c ih iw -> b (ih iw) c')
            k_y = rearrange(k_y, 'b c ih iw -> b (ih iw) c')
            q_y = rearrange(q_y, 'b n (h d) -> b h n d', h=self.heads)
            k_y = rearrange(k_y, 'b n (h d) -> b h n d', h=self.heads)
            dots = (dots + torch.matmul(q_y, k_y.transpose(-1, -2))) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        out = self.to_out(out)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=1, embed_dim=32, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        x = self.proj(x)

        _, _, H, W = x.shape
        # flatten: [B, C, D, H, W] -> [B, C, DHW]
        # transpose: [B, C, DHW] -> [B, DHW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()

        self.ih, self.iw = image_size
        self.layer1 = NoBottleneck(inp, inp, inp)

        self.attn = Attention(inp, inp, image_size, heads, dim_head, dropout)
        self.mlp = CMLP(inp, 4 * inp, drop=dropout)
        self.norm = nn.GroupNorm(1, inp, eps=1e-6)

        self.SA1 = SpatialAttention(7)

        self.conv1x1 = nn.Conv2d(2*inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, CONV, TRANS):
        CONV, x_totran = self.layer1(CONV)  # inp

        TRANS = self.norm(TRANS)
        TRANS = self.attn(TRANS, x_totran) + TRANS
        TRANS = self.norm(TRANS)
        TRANS = self.mlp(TRANS) + TRANS

        CONV_SA = self.SA1(TRANS) * CONV
        F = torch.cat([CONV_SA, TRANS], dim=1)
        CONV = self.conv1x1(F)
        return CONV, TRANS


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)  # Upsample
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Feature Concatenation
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PCTNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))

        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))

        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))

        self.s3 = makelayer(channels[3], channels[3], num_blocks[3], (ih // 16, iw // 16))

        self.upsamplex2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_more = Conv2dReLU(channels[3], channels[2], kernel_size=3, padding=1, use_batchnorm=True)

        self.ups2 = DecoderBlock(channels[2], channels[1], channels[2])
        self.ups3 = DecoderBlock(channels[1], channels[0], channels[1])
        self.ups4 = DecoderBlock(channels[0], channels[0], channels[0])

        self.heading = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv2dReLU(channels[0], channels[0] // 2, kernel_size=3, padding=1, use_batchnorm=True),
            nn.Conv2d(channels[0] // 2, 9, kernel_size=1, padding=0))
        self.T_down = PatchEmbed(
            patch_size=16, in_c=in_channels, embed_dim=channels[3],
            norm_layer=nn.LayerNorm)

        self.layer3 = nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, groups=1)

    def forward(self, input):
        # down
        input = input.repeat(1, 3, 1, 1)

        x = self.s0(input)
        skip1 = x

        x = self.s1(x)
        skip2 = x

        x = self.s2(x)
        skip3 = x

        x_t, H, W = self.T_down(input)
        x_t = rearrange(x_t, 'b (h w) c -> b c h w ', h=H, w=W)
        x = self.layer3(skip3)

        x, x_t = self.s3(x, x_t)

        x = self.conv_more(x+x_t)

        x = self.ups2(x, skip3)
        x = self.ups3(x, skip2)
        x = self.ups4(x, skip1)
        x = self.heading(x)

        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


class makelayer(nn.Module):
    def __init__(self, inp, oup, depth, image_size):
        super().__init__()

        self.strat = Transformer(inp, oup, image_size)

        self.blocks = nn.ModuleList([
            Transformer(oup, oup, image_size)
            for i in range(depth - 1)
        ])

    def forward(self, CONV, TRANS):

        CONV, TRANS = self.strat(CONV, TRANS)
        for blk in self.blocks:
            CONV, TRANS = blk(CONV, TRANS)
        return CONV, TRANS

def pctnet():
    num_blocks = [2, 2, 6, 6]
    channels = [64, 128, 256, 512]
    return PCTNet((224, 224), 3, num_blocks, channels)
