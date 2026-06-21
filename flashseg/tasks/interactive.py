"""Interactive segmentation using point/box/text prompts.

Provides a high-level interface on top of SAM for interactive segmentation
with iterative refinement.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PromptPoint:
    """A point prompt with foreground/background label."""

    x: float
    y: float
    is_foreground: bool = True


@dataclass
class PromptBox:
    """A bounding box prompt."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class InteractiveSession:
    """State for an interactive segmentation session on a single image."""

    image_embeddings: Optional[torch.Tensor] = None
    points: List[PromptPoint] = field(default_factory=list)
    boxes: List[PromptBox] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    mask_history: List[np.ndarray] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)

    def add_point(self, x: float, y: float, is_foreground: bool = True):
        self.points.append(PromptPoint(x, y, is_foreground))

    def add_box(self, x1: float, y1: float, x2: float, y2: float):
        self.boxes.append(PromptBox(x1, y1, x2, y2))

    def undo(self):
        """Remove the last prompt and revert to previous mask."""
        if self.points:
            self.points.pop()
        elif self.boxes:
            self.boxes.pop()
        if self.mask_history:
            self.current_mask = self.mask_history.pop()
        else:
            self.current_mask = None

    def clear(self):
        self.points.clear()
        self.boxes.clear()
        self.current_mask = None
        self.mask_history.clear()


class InteractiveSegmentor:
    """Interactive segmentation with point/box/text prompts using SAM.

    Manages sessions that cache image embeddings for efficient
    iterative prompt-based refinement.

    Args:
        model: A SAM model instance (from ``flashseg.models.architectures.sam``).
        device: Torch device.
        mask_threshold: Threshold for converting logits to binary mask.
        max_points: Maximum number of points per session.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        mask_threshold: float = 0.0,
        max_points: int = 64,
    ):
        self.model = model
        self.device = torch.device(device)
        self.mask_threshold = mask_threshold
        self.max_points = max_points
        self._sessions: Dict[str, InteractiveSession] = {}

    def create_session(
        self,
        session_id: str,
        image: torch.Tensor,
    ) -> InteractiveSession:
        """Initialize a new interactive session for an image.

        Pre-computes image embeddings so subsequent prompts are fast.

        Args:
            session_id: Unique identifier for this session.
            image: (1, 3, H, W) input image tensor.

        Returns:
            New ``InteractiveSession``.
        """
        self.model.eval()
        image = image.to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_image_embeddings(image)

        session = InteractiveSession(
            image_embeddings=embeddings,
            image_size=(image.shape[2], image.shape[3]),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[InteractiveSession]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def add_point_and_predict(
        self,
        session_id: str,
        x: float,
        y: float,
        is_foreground: bool = True,
    ) -> np.ndarray:
        """Add a point prompt and predict a new mask.

        Args:
            session_id: Session identifier.
            x: Point x-coordinate.
            y: Point y-coordinate.
            is_foreground: True for foreground, False for background.

        Returns:
            (H, W) binary mask as numpy array.
        """
        session = self._sessions[session_id]
        session.add_point(x, y, is_foreground)

        if session.current_mask is not None:
            session.mask_history.append(session.current_mask.copy())

        mask = self._predict_from_session(session)
        session.current_mask = mask
        return mask

    def add_box_and_predict(
        self,
        session_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        """Add a box prompt and predict a new mask."""
        session = self._sessions[session_id]
        session.add_box(x1, y1, x2, y2)

        if session.current_mask is not None:
            session.mask_history.append(session.current_mask.copy())

        mask = self._predict_from_session(session)
        session.current_mask = mask
        return mask

    def predict(
        self,
        session_id: str,
        points: Optional[List[Tuple[float, float, bool]]] = None,
        boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> np.ndarray:
        """Predict a mask with explicit prompts (does not modify session state).

        Args:
            session_id: Session identifier.
            points: List of (x, y, is_foreground) tuples.
            boxes: List of (x1, y1, x2, y2) tuples.

        Returns:
            (H, W) binary mask.
        """
        session = self._sessions[session_id]

        temp_session = InteractiveSession(
            image_embeddings=session.image_embeddings,
            image_size=session.image_size,
        )
        if points:
            for x, y, fg in points:
                temp_session.add_point(x, y, fg)
        if boxes:
            for x1, y1, x2, y2 in boxes:
                temp_session.add_box(x1, y1, x2, y2)

        return self._predict_from_session(temp_session)

    def undo(self, session_id: str) -> Optional[np.ndarray]:
        """Undo the last prompt in a session."""
        session = self._sessions[session_id]
        session.undo()
        return session.current_mask

    @torch.no_grad()
    def _predict_from_session(self, session: InteractiveSession) -> np.ndarray:
        """Run SAM mask prediction using session prompts."""
        device = self.device

        point_coords = None
        point_labels = None
        if session.points:
            coords = [[p.x, p.y] for p in session.points[-self.max_points:]]
            labels = [1 if p.is_foreground else 0 for p in session.points[-self.max_points:]]
            point_coords = torch.tensor([coords], dtype=torch.float32, device=device)
            point_labels = torch.tensor([labels], dtype=torch.long, device=device)

        box_tensor = None
        if session.boxes:
            last_box = session.boxes[-1]
            box_tensor = torch.tensor(
                [[last_box.x1, last_box.y1, last_box.x2, last_box.y2]],
                dtype=torch.float32,
                device=device,
            )

        points = (point_coords, point_labels) if point_coords is not None else None

        sparse, dense = self.model.prompt_encoder(
            points=points, boxes=box_tensor,
        )

        pred_masks, iou_scores = self.model.mask_decoder(
            session.image_embeddings,
            sparse, dense,
            image_size=session.image_size,
        )

        best_idx = iou_scores.argmax(dim=1)
        B = pred_masks.shape[0]
        best_mask = pred_masks[torch.arange(B), best_idx]  # (B, H, W)

        binary = (best_mask > self.mask_threshold).squeeze(0).cpu().numpy().astype(np.uint8)
        return binary

    def auto_segment(
        self,
        image: torch.Tensor,
        points_per_side: int = 32,
        score_threshold: float = 0.5,
    ) -> List[np.ndarray]:
        """Automatic mask generation by sampling a grid of point prompts.

        Args:
            image: (1, 3, H, W) input image.
            points_per_side: Grid density for automatic point sampling.
            score_threshold: Minimum IoU score for keeping masks.

        Returns:
            List of (H, W) binary masks.
        """
        self.model.eval()
        image = image.to(self.device)
        H, W = image.shape[2], image.shape[3]

        with torch.no_grad():
            embeddings = self.model.get_image_embeddings(image)

        step_h = H / points_per_side
        step_w = W / points_per_side

        all_masks = []

        for i in range(points_per_side):
            for j in range(points_per_side):
                px = step_w * (j + 0.5)
                py = step_h * (i + 0.5)

                coords = torch.tensor([[[px, py]]], dtype=torch.float32, device=self.device)
                labels = torch.tensor([[1]], dtype=torch.long, device=self.device)

                sparse, dense = self.model.prompt_encoder(
                    points=(coords, labels),
                )
                pred_masks, iou_scores = self.model.mask_decoder(
                    embeddings, sparse, dense, image_size=(H, W),
                )

                best_idx = iou_scores.argmax(dim=1)
                best_score = iou_scores[0, best_idx[0]].item()
                if best_score < score_threshold:
                    continue

                mask = (pred_masks[0, best_idx[0]] > self.mask_threshold).cpu().numpy().astype(np.uint8)
                if mask.sum() > 0:
                    all_masks.append(mask)

        return _remove_duplicate_masks(all_masks)


def _remove_duplicate_masks(
    masks: List[np.ndarray],
    iou_threshold: float = 0.8,
) -> List[np.ndarray]:
    """Remove near-duplicate masks by IoU."""
    if not masks:
        return []

    keep = []
    for mask in masks:
        is_dup = False
        for kept in keep:
            intersection = (mask & kept).sum()
            union = (mask | kept).sum()
            if union > 0 and intersection / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(mask)
    return keep
