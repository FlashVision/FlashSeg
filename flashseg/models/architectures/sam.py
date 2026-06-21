"""Segment Anything Model (SAM).

Wraps SAM with image encoder, prompt encoder, and mask decoder.
Supports point, box, and text prompt-based segmentation.

References:
    Kirillov et al., "Segment Anything", ICCV 2023.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HF_SAM_AVAILABLE = False
try:
    from transformers import SamModel, SamProcessor
    _HF_SAM_AVAILABLE = True
except ImportError:
    pass


class PatchEmbed(nn.Module):
    """Image to patch embedding with 2D positional encoding."""

    def __init__(self, img_size: int = 1024, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Multi-head attention with optional window partitioning."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class ViTBlock(nn.Module):
    """ViT transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 window_size: int = 0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SAMImageEncoder(nn.Module):
    """SAM ViT-based image encoder.

    Args:
        img_size: Input image size (square).
        patch_size: Patch tokenization size.
        in_channels: Number of input channels.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        out_channels: Output feature channels (projected from embed_dim).
        window_size: Window attention size (0 for global attention).
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 256,
        window_size: int = 14,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        global_interval = max(depth // 4, 1)
        self.blocks = nn.ModuleList([
            ViTBlock(
                embed_dim, num_heads, mlp_ratio,
                window_size=0 if (i + 1) % global_interval == 0 else window_size,
            )
            for i in range(depth)
        ])
        self.neck = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image into spatial features.

        Args:
            x: (B, 3, H, W) input image.

        Returns:
            (B, H'*W', out_channels) spatial features.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x)
        return x


class SAMPromptEncoder(nn.Module):
    """Encodes point, box, and mask prompts into embeddings.

    Args:
        embed_dim: Prompt embedding dimension.
        img_size: Image size for coordinate normalization.
        mask_in_channels: Input channels for mask prompts.
    """

    def __init__(self, embed_dim: int = 256, img_size: int = 1024, mask_in_channels: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.point_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.label_embed = nn.Embedding(2, embed_dim)  # 0 = background, 1 = foreground

        self.box_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_in_channels, embed_dim // 4, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 2, 1),
            nn.GELU(),
        )

        self.no_prompt_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts.

        Args:
            points: Tuple of (coords (B, N, 2), labels (B, N)) for point prompts.
            boxes: (B, 4) bounding box prompts [x1, y1, x2, y2].
            masks: (B, 1, H, W) mask prompts.

        Returns:
            (sparse_embeddings, dense_embeddings):
              sparse: (B, N_prompts, embed_dim)
              dense: (B, embed_dim, H', W') or None.
        """
        B = 1
        sparse_parts = []

        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            coords_norm = coords / self.img_size
            pt_embed = self.point_embed(coords_norm) + self.label_embed(labels)
            sparse_parts.append(pt_embed)

        if boxes is not None:
            B = boxes.shape[0]
            boxes_norm = boxes / self.img_size
            bx_embed = self.box_embed(boxes_norm).unsqueeze(1)
            sparse_parts.append(bx_embed)

        if sparse_parts:
            sparse = torch.cat(sparse_parts, dim=1)
        else:
            sparse = self.no_prompt_embed.weight.unsqueeze(0).expand(B, -1, -1)

        dense = None
        if masks is not None:
            dense = self.mask_conv(masks)

        return sparse, dense


class SAMMaskDecoder(nn.Module):
    """Lightweight mask decoder that produces masks from image + prompt embeddings.

    Uses cross-attention between prompt tokens and image features, followed
    by a transposed-convolution upsampling head.

    Args:
        embed_dim: Feature embedding dimension.
        num_heads: Number of attention heads in cross-attention.
        num_mask_tokens: Number of output mask tokens (for multi-mask prediction).
        depth: Number of transformer decoder layers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_mask_tokens: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mask_tokens = num_mask_tokens

        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)
        self.iou_token = nn.Embedding(1, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.GELU(),
        )

        self.mask_heads = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // 8) for _ in range(num_mask_tokens)
        ])
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_mask_tokens),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt: torch.Tensor,
        dense_prompt: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (1024, 1024),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masks from image + prompt embeddings.

        Args:
            image_embeddings: (B, N, embed_dim) from image encoder.
            sparse_prompt: (B, P, embed_dim) from prompt encoder.
            dense_prompt: Optional (B, embed_dim, H', W') dense prompt.
            image_size: Target output mask size (H, W).

        Returns:
            masks: (B, num_mask_tokens, H, W) predicted masks.
            iou_scores: (B, num_mask_tokens) predicted mask IoU scores.
        """
        B = image_embeddings.shape[0]

        mask_tok = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        iou_tok = self.iou_token.weight.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([iou_tok, mask_tok, sparse_prompt], dim=1)

        decoded = self.decoder(tokens, image_embeddings)

        iou_out = decoded[:, 0]
        mask_out = decoded[:, 1: 1 + self.num_mask_tokens]

        iou_scores = self.iou_head(iou_out)

        N = image_embeddings.shape[1]
        h = w = int(math.sqrt(N))
        img_feat_2d = image_embeddings.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        if dense_prompt is not None:
            dp_resized = F.interpolate(dense_prompt, size=(h, w), mode="bilinear", align_corners=False)
            img_feat_2d = img_feat_2d + dp_resized

        upscaled = self.upscale(img_feat_2d)
        _, cup, hu, wu = upscaled.shape

        masks = []
        for i in range(self.num_mask_tokens):
            proj = self.mask_heads[i](mask_out[:, i])  # (B, cup)
            proj = proj.unsqueeze(-1).unsqueeze(-1)
            mask = (upscaled * proj).sum(dim=1, keepdim=True)
            mask = F.interpolate(mask, size=image_size, mode="bilinear", align_corners=False)
            masks.append(mask)

        masks = torch.cat(masks, dim=1)
        return masks, iou_scores


class SAM(nn.Module):
    """Segment Anything Model — unified promptable segmentation.

    Supports two modes:
      1. **Standalone**: Built-in ViT encoder + prompt encoder + mask decoder.
      2. **HuggingFace**: Wraps a pretrained SAM model from HuggingFace.

    Args:
        img_size: Input image size (square).
        patch_size: ViT patch size.
        embed_dim: ViT embedding dimension.
        encoder_depth: Number of ViT blocks.
        num_heads: Attention heads.
        decoder_embed_dim: Mask decoder embedding dim.
        num_mask_tokens: Number of predicted masks per prompt.
        hf_model_name: HuggingFace model name (e.g. ``"facebook/sam-vit-base"``).
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 256,
        num_mask_tokens: int = 4,
        hf_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size

        self.hf_mode = hf_model_name is not None and _HF_SAM_AVAILABLE
        if self.hf_mode:
            self.hf_model = SamModel.from_pretrained(hf_model_name)
            self.hf_processor = SamProcessor.from_pretrained(hf_model_name)
            self.image_encoder = None
            self.prompt_encoder = None
            self.mask_decoder = None
        else:
            self.hf_model = None
            self.hf_processor = None
            self.image_encoder = SAMImageEncoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=encoder_depth,
                num_heads=num_heads,
                out_channels=decoder_embed_dim,
            )
            self.prompt_encoder = SAMPromptEncoder(
                embed_dim=decoder_embed_dim,
                img_size=img_size,
            )
            self.mask_decoder = SAMMaskDecoder(
                embed_dim=decoder_embed_dim,
                num_mask_tokens=num_mask_tokens,
            )

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with prompt-based segmentation.

        Args:
            images: (B, 3, H, W) input images.
            points: Tuple of (coords (B, N, 2), labels (B, N)).
            boxes: (B, 4) bounding box prompts.
            masks: (B, 1, H, W) mask prompts.

        Returns:
            Dict with ``'masks'`` (B, num_tokens, H, W) and ``'iou_scores'``.
        """
        if self.hf_mode:
            return self._forward_hf(images, points, boxes)

        image_embeddings = self.image_encoder(images)
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=masks)
        pred_masks, iou_scores = self.mask_decoder(
            image_embeddings, sparse, dense,
            image_size=(images.shape[2], images.shape[3]),
        )
        return {"masks": pred_masks, "iou_scores": iou_scores}

    def _forward_hf(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"pixel_values": images}
        if points is not None:
            kwargs["input_points"] = points[0]
            kwargs["input_labels"] = points[1]
        if boxes is not None:
            kwargs["input_boxes"] = boxes.unsqueeze(1)
        outputs = self.hf_model(**kwargs)
        return {
            "masks": outputs.pred_masks,
            "iou_scores": outputs.iou_scores,
        }

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Inference with automatic best-mask selection.

        Args:
            image: (1, 3, H, W) single image.
            points: Point prompts.
            boxes: Box prompts.
            multimask: If False, returns only the highest-IoU mask.

        Returns:
            Dict with ``'masks'`` and ``'iou_scores'``.
        """
        self.eval()
        result = self.forward(image, points=points, boxes=boxes)

        if not multimask:
            best_idx = result["iou_scores"].argmax(dim=1)
            B = result["masks"].shape[0]
            best_masks = result["masks"][torch.arange(B), best_idx].unsqueeze(1)
            best_scores = result["iou_scores"][torch.arange(B), best_idx].unsqueeze(1)
            result = {"masks": best_masks, "iou_scores": best_scores}

        return result

    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Pre-compute image embeddings for efficient multi-prompt inference."""
        if self.hf_mode:
            return self.hf_model.get_image_embeddings(images)
        return self.image_encoder(images)
