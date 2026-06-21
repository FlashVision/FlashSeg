"""SAM2 — Segment Anything Model 2 for video segmentation.

Extends SAM with memory attention and temporal propagation for tracking
and segmenting objects across video frames.

References:
    Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam import SAMImageEncoder, SAMPromptEncoder, SAMMaskDecoder

logger = logging.getLogger(__name__)


class MemoryAttention(nn.Module):
    """Cross-attention module that attends to stored memory features.

    Allows current-frame features to attend to features from previously
    processed frames, enabling temporal reasoning.

    Args:
        embed_dim: Feature embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
                "norm3": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(
        self,
        current_features: torch.Tensor,
        memory_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse current frame features with memory.

        Args:
            current_features: (B, N, D) features of the current frame.
            memory_features: (B, M, D) stored memory features.

        Returns:
            (B, N, D) updated features.
        """
        x = current_features
        for layer in self.layers:
            res = x
            x = layer["norm1"](x)
            x, _ = layer["self_attn"](x, x, x)
            x = x + res

            res = x
            x = layer["norm2"](x)
            x, _ = layer["cross_attn"](x, memory_features, memory_features)
            x = x + res

            res = x
            x = layer["norm3"](x)
            x = layer["ffn"](x)
            x = x + res

        return x


class MemoryEncoder(nn.Module):
    """Encodes frame features + mask into compact memory tokens.

    Args:
        embed_dim: Feature embedding dimension.
        num_tokens: Number of memory tokens per frame.
    """

    def __init__(self, embed_dim: int = 256, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 2, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.compress = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create memory tokens from image features and predicted mask.

        Args:
            image_features: (B, N, D) image features.
            mask: (B, 1, H, W) predicted mask.

        Returns:
            (B, num_tokens, D) memory tokens.
        """
        B, N, D = image_features.shape
        h = w = int(N ** 0.5)

        mask_feat = self.mask_proj(mask)
        mask_feat = F.adaptive_avg_pool2d(mask_feat, (h, w))
        mask_feat = mask_feat.flatten(2).transpose(1, 2)  # (B, N, D)

        fused = self.fuse(torch.cat([image_features, mask_feat], dim=-1))

        if N > self.num_tokens:
            fused = fused[:, :self.num_tokens]
        elif N < self.num_tokens:
            pad = torch.zeros(B, self.num_tokens - N, D, device=fused.device)
            fused = torch.cat([fused, pad], dim=1)

        return self.compress(fused)


class TemporalPropagator(nn.Module):
    """Propagates object masks across video frames using memory bank.

    Maintains a fixed-size memory bank of recent frame features and masks,
    using memory attention to condition current-frame predictions.

    Args:
        embed_dim: Feature dimension.
        memory_size: Maximum number of frames in memory.
        num_memory_tokens: Tokens stored per frame.
        num_heads: Attention heads for memory attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        memory_size: int = 6,
        num_memory_tokens: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_attention = MemoryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.memory_encoder = MemoryEncoder(
            embed_dim=embed_dim,
            num_tokens=num_memory_tokens,
        )
        self._memory_bank: List[torch.Tensor] = []

    def reset(self):
        """Clear the memory bank."""
        self._memory_bank = []

    def add_to_memory(
        self,
        image_features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Store a frame's features in the memory bank.

        Args:
            image_features: (B, N, D) image features.
            mask: (B, 1, H, W) predicted mask for this frame.
        """
        mem_tokens = self.memory_encoder(image_features, mask)
        self._memory_bank.append(mem_tokens.detach())
        if len(self._memory_bank) > self.memory_size:
            self._memory_bank.pop(0)

    def propagate(
        self,
        current_features: torch.Tensor,
    ) -> torch.Tensor:
        """Condition current frame features on memory.

        Args:
            current_features: (B, N, D) features of the current frame.

        Returns:
            (B, N, D) memory-conditioned features.
        """
        if not self._memory_bank:
            return current_features

        memory = torch.cat(self._memory_bank, dim=1)
        return self.memory_attention(current_features, memory)


class SAM2(nn.Module):
    """SAM2 — Segment Anything Model 2 for image and video segmentation.

    Extends SAM with a memory mechanism for video object segmentation:
    features from previously segmented frames are stored and used to
    condition predictions on new frames via cross-attention.

    Args:
        img_size: Input image size.
        patch_size: ViT patch size.
        embed_dim: ViT embedding dimension.
        encoder_depth: Number of ViT encoder blocks.
        num_heads: Attention heads.
        decoder_embed_dim: Mask decoder embedding dimension.
        num_mask_tokens: Predicted masks per prompt.
        memory_size: Number of frames stored in memory.
        memory_tokens: Tokens per memory frame.
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
        memory_size: int = 6,
        memory_tokens: int = 64,
    ):
        super().__init__()
        self.img_size = img_size

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
        self.temporal = TemporalPropagator(
            embed_dim=decoder_embed_dim,
            memory_size=memory_size,
            num_memory_tokens=memory_tokens,
            num_heads=num_heads,
        )

    def reset_memory(self):
        """Clear temporal memory (call between videos)."""
        self.temporal.reset()

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        use_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Segment a single frame, optionally using temporal memory.

        Args:
            images: (B, 3, H, W) input frame.
            points: Point prompts.
            boxes: Box prompts.
            masks: Mask prompts.
            use_memory: Whether to use/update temporal memory.

        Returns:
            Dict with ``'masks'``, ``'iou_scores'``.
        """
        image_features = self.image_encoder(images)

        if use_memory:
            image_features = self.temporal.propagate(image_features)

        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=masks)
        pred_masks, iou_scores = self.mask_decoder(
            image_features, sparse, dense,
            image_size=(images.shape[2], images.shape[3]),
        )

        if use_memory:
            best_idx = iou_scores.argmax(dim=1)
            B = pred_masks.shape[0]
            best_mask = pred_masks[torch.arange(B), best_idx].unsqueeze(1)
            raw_features = self.image_encoder(images)
            self.temporal.add_to_memory(raw_features, (best_mask > 0).float())

        return {"masks": pred_masks, "iou_scores": iou_scores}

    @torch.no_grad()
    def track_video(
        self,
        frames: List[torch.Tensor],
        init_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        init_boxes: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Track and segment an object across a sequence of frames.

        The first frame uses the provided prompts; subsequent frames propagate
        via temporal memory without additional prompts.

        Args:
            frames: List of (1, 3, H, W) frame tensors.
            init_points: Point prompts for the first frame.
            init_boxes: Box prompts for the first frame.

        Returns:
            List of result dicts per frame.
        """
        self.eval()
        self.reset_memory()

        results = []
        for i, frame in enumerate(frames):
            if i == 0:
                result = self.forward(
                    frame, points=init_points, boxes=init_boxes, use_memory=True,
                )
            else:
                result = self.forward(frame, use_memory=True)
            results.append(result)

        return results
