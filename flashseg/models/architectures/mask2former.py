"""Mask2Former — Masked-attention transformer for unified segmentation.

Implements the Mask2Former architecture with masked attention in the
transformer decoder for semantic, instance, and panoptic segmentation.

References:
    Cheng et al., "Masked-attention Mask Transformer for Universal
    Image Segmentation", CVPR 2022.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PixelDecoder(nn.Module):
    """Multi-scale feature pyramid pixel decoder.

    Takes multi-scale backbone features and produces per-pixel embeddings
    at 1/4 resolution via lateral connections and top-down fusion.

    Args:
        in_channels_list: Channel dimensions from each backbone stage.
        out_channels: Output feature dimension.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [64, 128, 256],
        out_channels: int = 256,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.mask_features = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(
        self, features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Produce per-pixel embeddings and multi-scale features.

        Args:
            features: List of (B, C_i, H_i, W_i) backbone features, low-to-high res.

        Returns:
            mask_features: (B, out_channels, H0, W0) high-res pixel embeddings.
            multi_scale: List of multi-scale feature maps for cross-attention.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i] = laterals[i] + up

        multi_scale = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        mask_features = self.mask_features(multi_scale[0])

        return mask_features, multi_scale


class MaskedCrossAttention(nn.Module):
    """Cross-attention with predicted mask attention bias.

    Each query only attends to the spatial region predicted by its mask
    from the previous layer, focusing computation on relevant areas.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, D) object queries.
            key: (B, N, D) spatial features.
            value: (B, N, D) spatial features.
            attn_mask: (B, Q, N) boolean attention mask from predicted masks.

        Returns:
            (B, Q, D) attended features.
        """
        B, Q, D = query.shape
        N = key.shape[1]

        q = self.q_proj(query).reshape(B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # (B, Q, N) -> (B, 1, Q, N) -> broadcast to (B, num_heads, Q, N)
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Q, D)

        return self.out_proj(out)


class MaskedTransformerDecoderLayer(nn.Module):
    """Single Mask2Former decoder layer with masked cross-attention."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = MaskedCrossAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, Q, D) object queries.
            features: (B, N, D) spatial features.
            attn_mask: (B, Q, N) boolean mask attention bias.

        Returns:
            (B, Q, D) updated queries.
        """
        q = self.norm1(queries)
        q, _ = self.self_attn(q, q, q)
        queries = queries + q

        q = self.norm2(queries)
        q = self.cross_attn(q, features, features, attn_mask=attn_mask)
        queries = queries + q

        queries = queries + self.ffn(self.norm3(queries))
        return queries


class MaskedTransformerDecoder(nn.Module):
    """Mask2Former transformer decoder with iterative mask prediction.

    Uses object queries that iteratively refine mask predictions via
    masked cross-attention.

    Args:
        num_classes: Number of semantic categories.
        embed_dim: Query/feature embedding dimension.
        num_heads: Attention heads.
        num_layers: Number of decoder layers.
        num_queries: Number of object queries.
        ffn_dim: FFN hidden dimension.
    """

    def __init__(
        self,
        num_classes: int = 21,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_queries: int = 100,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_feat = nn.Embedding(num_queries, embed_dim)

        self.layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.class_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes + 1)
            for _ in range(num_layers)
        ])
        self.mask_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_layers)
        ])

        self.level_embed = nn.Embedding(3, embed_dim)

    def forward(
        self,
        mask_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run iterative mask prediction.

        Args:
            mask_features: (B, D, H, W) high-res pixel embeddings.
            multi_scale_features: List of (B, D, Hi, Wi) multi-scale features.

        Returns:
            Dict with:
              ``'pred_logits'``: (B, Q, num_classes+1) class predictions.
              ``'pred_masks'``: (B, Q, H, W) mask predictions.
              ``'aux_outputs'``: List of intermediate predictions.
        """
        B = mask_features.shape[0]

        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)

        all_class_preds = []
        all_mask_preds = []

        for layer_idx, layer in enumerate(self.layers):
            scale_idx = layer_idx % len(multi_scale_features)
            feat = multi_scale_features[scale_idx]
            _, _, fh, fw = feat.shape
            spatial = feat.flatten(2).transpose(1, 2)  # (B, N, D)
            spatial = spatial + self.level_embed.weight[scale_idx].unsqueeze(0).unsqueeze(0)

            if layer_idx > 0:
                prev_masks = all_mask_preds[-1]  # (B, Q, H, W)
                mask_down = F.interpolate(
                    prev_masks, size=(fh, fw), mode="bilinear", align_corners=False,
                )
                attn_mask = (mask_down.flatten(2) < 0.0)  # (B, Q, N)
            else:
                attn_mask = None

            queries = layer(queries, spatial, attn_mask=attn_mask)

            class_pred = self.class_heads[layer_idx](queries)
            mask_embed = self.mask_heads[layer_idx](queries)

            _, D, mh, mw = mask_features.shape
            mask_pred = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)

            all_class_preds.append(class_pred)
            all_mask_preds.append(mask_pred)

        aux_outputs = [
            {"pred_logits": c, "pred_masks": m}
            for c, m in zip(all_class_preds[:-1], all_mask_preds[:-1])
        ]

        return {
            "pred_logits": all_class_preds[-1],
            "pred_masks": all_mask_preds[-1],
            "aux_outputs": aux_outputs,
        }


class Mask2FormerBackbone(nn.Module):
    """Simple multi-scale backbone for Mask2Former."""

    def __init__(self, in_channels: int = 3, channels: List[int] = [64, 128, 256]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))
            prev_ch = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class Mask2Former(nn.Module):
    """Mask2Former — unified architecture for semantic, instance, and panoptic segmentation.

    Args:
        num_classes: Number of semantic categories.
        backbone_channels: Channel dimensions per backbone stage.
        embed_dim: Transformer embedding dimension.
        num_heads: Attention heads.
        num_layers: Decoder layers.
        num_queries: Object queries.
    """

    def __init__(
        self,
        num_classes: int = 21,
        backbone_channels: List[int] = [64, 128, 256],
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_queries: int = 100,
    ):
        super().__init__()
        self.backbone = Mask2FormerBackbone(channels=backbone_channels)
        self.pixel_decoder = PixelDecoder(
            in_channels_list=backbone_channels,
            out_channels=embed_dim,
        )
        self.transformer_decoder = MaskedTransformerDecoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_queries=num_queries,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image.

        Returns:
            Dict with ``'pred_logits'``, ``'pred_masks'``, ``'aux_outputs'``.
        """
        features = self.backbone(x)
        mask_features, multi_scale = self.pixel_decoder(features)
        return self.transformer_decoder(mask_features, multi_scale)

    @torch.no_grad()
    def predict_semantic(self, x: torch.Tensor) -> torch.Tensor:
        """Predict semantic segmentation map.

        Args:
            x: (B, 3, H, W) input.

        Returns:
            (B, H, W) class index map.
        """
        self.eval()
        out = self.forward(x)
        pred_logits = out["pred_logits"]  # (B, Q, C+1)
        pred_masks = out["pred_masks"]  # (B, Q, H', W')

        pred_masks = F.interpolate(
            pred_masks, size=x.shape[2:], mode="bilinear", align_corners=False,
        )

        cls_probs = pred_logits.softmax(-1)[..., :-1]  # drop no-object
        mask_probs = pred_masks.sigmoid()

        sem_seg = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)
        return sem_seg.argmax(dim=1)

    @torch.no_grad()
    def predict_instance(self, x: torch.Tensor, score_threshold: float = 0.5):
        """Predict instance segmentation.

        Returns:
            List of dicts per image: each dict has ``'labels'``, ``'scores'``,
            ``'masks'`` (N, H, W).
        """
        self.eval()
        out = self.forward(x)
        pred_logits = out["pred_logits"]
        pred_masks = out["pred_masks"]

        pred_masks = F.interpolate(
            pred_masks, size=x.shape[2:], mode="bilinear", align_corners=False,
        )

        results = []
        B = pred_logits.shape[0]
        for b in range(B):
            scores, labels = pred_logits[b].softmax(-1)[..., :-1].max(-1)
            keep = scores > score_threshold
            results.append({
                "labels": labels[keep],
                "scores": scores[keep],
                "masks": (pred_masks[b][keep] > 0).float(),
            })
        return results
