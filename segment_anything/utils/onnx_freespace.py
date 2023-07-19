import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple
from ..modeling import ImageEncoderViT
import einops    

class SamVit4RoadSegOnnx(nn.Module):
    def __init__(
        self,
        vit_model: ImageEncoderViT
    ) -> None:
        super().__init__()
        self.vit_model = vit_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit_model.patch_embed(x)
        if self.vit_model.pos_embed is not None:
            x = x + self.vit_model.pos_embed 

        out = []
        for blk in self.vit_model.blocks:
            x = blk(x)
            out.append(x)

        x = self.vit_model.neck(x.permute(0, 3, 1, 2))
        out = torch.stack(out, axis=0)
        return x, out


class ConvModule(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, kernel_size=0, padding=0, stride=1):
        super(ConvModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            in_channels=[1280]*32,
            inner_channels=128,
            selected_channels = range(4, 32, 2),
            out_channels=256,
            up_sample_scale=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.selected_channels = selected_channels
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
            )
        )

        self.up_sample_layers = nn.ModuleList()
        assert up_sample_scale == 4
        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, image_embedding, inner_states):
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0, image_embedding


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x


class SegHead(nn.Module):
    def __init__(self,):
        super(SegHead, self).__init__()
        self.in_channels = [256, 256, 256, 256]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 256

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_ch=embedding_dim * 4,
            out_ch=embedding_dim,
            kernel_size=1,
        )

        self.neck_net = SAMAggregatorNeck()
        # self.linear_pred = nn.Sequential(
        #     nn.ConvTranspose2d(embedding_dim, embedding_dim // 2, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(embedding_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(embedding_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embedding_dim // 2, 2, kernel_size=1)
        # )

        self.linear_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)

    def forward(self, image_embeddings, inner_state):
        x = self.neck_net(image_embeddings, inner_state)

        c1, c2, c3, c4 = x

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(x)

        return x

class SegHeadOnnx(nn.Module):
    def __init__(
        self, 
        model: SegHead
    ):
        super().__init__()
        self.model = model
        self.img_size = 1024
        
    def resize_longest_image_size(self,
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size
        
    def mask_postprocessing(self, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(torch.int64)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

        orig_im_size = orig_im_size.to(torch.int32)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks
    
    def forward(
        self, 
        image_embeddings: torch.Tensor,
        inner_state: torch.Tensor,
        ori_size: torch.Tensor
    ):
        pred_mask = self.model(image_embeddings, inner_state)
        pred_mask = self.mask_postprocessing(pred_mask, ori_size)
        return pred_mask